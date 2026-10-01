"""Execution on Prim: the model is HANDED the expert-verified gold primitive and solves the problem.

Protocol (as in the paper):
  * prompt = prompts.EXECUTION_PROMPT, single user message, no system prompt. The dataset's
    primitive fields fill the prompt slots.
  * reasoning_effort="high", max_completion_tokens=120000; serve with --max-model-len 131072
  * the answer is read from the `## Final Answer` block.
The response is graded afterwards against `answer` by eval/judge_answer.py (the official
HLE judge, o3-mini).

Example:
    python eval/execution.py --model qwen-3.5-9b-absorb --base-url http://localhost:8000/v1
Resume is automatic: re-running fills in only missing / failed rows.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import JsonlWriter, base_urls, extract_final_answer_block, load_prim, usage_dict  # noqa: E402
from prompts import EXECUTION_PROMPT  # noqa: E402


def build_prompt(it: dict) -> str:
    p = it["primitive"]
    return EXECUTION_PROMPT.format(problem=it["question"], essential_property=p["essential_property"],
                                   solution_principle=p["solution_principle"], core_concept=p["core_concept"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible base URL(s), comma-separated")
    ap.add_argument("--out", default=None, help="default: results/<model>/execution.jsonl")
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--max-tokens", type=int, default=120000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    items = load_prim()[: args.limit]
    out = JsonlWriter(args.out or f"results/{args.model}/execution.jsonl")
    done = {r["idx"] for r in out.load() if (r.get("response") or "").strip()}
    todo = [it for it in items if it["idx"] not in done]
    urls = base_urls(args.base_url)
    clients = [OpenAI(base_url=u, api_key=os.environ.get("PRIM_API_KEY", "EMPTY"),
                      timeout=10800.0, max_retries=2) for u in urls]
    print(f"[execution] model={args.model} rows={len(items)} max_tokens={args.max_tokens} "
          f"todo={len(todo)} (done {len(done)}) -> {out.path}\n"
          f"[execution] serve the model with --max-model-len 131072 (no --reasoning-parser)", flush=True)

    def run(job):
        k, it = job
        t0 = time.time()
        kwargs = dict(model=args.model, max_completion_tokens=args.max_tokens,
                      messages=[{"role": "user", "content": build_prompt(it)}])
        if args.reasoning_effort:
            kwargs["reasoning_effort"] = args.reasoning_effort
        resp = clients[k % len(clients)].chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        return {"idx": it["idx"], "model": args.model, "question": it["question"], "answer": it["answer"],
                "response": text, "pred": extract_final_answer_block(text),
                "finish_reason": choice.finish_reason, "usage": usage_dict(resp),
                "elapsed": round(time.time() - t0, 1)}

    n_ok = n_err = 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run, job): job[1] for job in enumerate(todo)}
        for fut in cf.as_completed(futs):
            it = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:  # not written -> retried on the next run
                n_err += 1
                print(f"FAIL {it['idx']}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            if not rec["response"].strip():
                n_err += 1
                print(f"EMPTY {it['idx']}; will retry on rerun", flush=True)
                continue
            out.append(rec)
            n_ok += 1
            if n_ok % 10 == 0 or n_ok + n_err == len(todo):
                print(f"[execution] {n_ok + n_err}/{len(todo)} written={n_ok} failed={n_err}", flush=True)

    recs = out.load()
    print(f"[execution] DONE: {len(recs)}/{len(items)} rows on disk, "
          f"{sum(1 for r in recs if r.get('pred'))} with a Final Answer block, "
          f"{sum(1 for r in recs if r.get('finish_reason') == 'length')} hit the token cap, "
          f"{n_err} failed this run (re-run to retry).", flush=True)


if __name__ == "__main__":
    main()
