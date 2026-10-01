"""Generation on Prim: the model answers each problem directly, with NO primitive.

Protocol (frozen, as in the paper):
  * system = HLE's answer-format prompt, user = the problem
  * max_completion_tokens=120000, one draw per problem, sampling parameters left to the
    server (i.e. the model's generation_config.json, or vLLM defaults if it has none)
  * serve with --max-model-len 131072 and no --reasoning-parser
The response is graded afterwards against `answer` by eval/judge_answer.py (the official
HLE judge, o3-mini).

Example:
    python eval/generation.py --model qwen-3.5-9b-absorb --base-url http://localhost:8000/v1
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
from common import HLE_SYSTEM_PROMPT, JsonlWriter, base_urls, extract_answer_line, load_prim, usage_dict  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="served model name (vllm --served-model-name)")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible base URL(s), comma-separated for round-robin "
                         "(default: $PRIM_BASE_URL or http://localhost:8000/v1)")
    ap.add_argument("--out", default=None, help="output JSONL (default: results/<model>/generation.jsonl)")
    ap.add_argument("--max-tokens", type=int, default=120000)
    ap.add_argument("--draws", type=int, default=1, help="samples per problem (paper: 1)")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="only the first N problems (smoke test)")
    args = ap.parse_args()

    items = load_prim()[: args.limit]
    out = JsonlWriter(args.out or f"results/{args.model}/generation.jsonl")
    done = {(r["idx"], r["draw"]) for r in out.load() if (r.get("response") or "").strip() and not r.get("error")}
    todo = [(it, d) for it in items for d in range(args.draws) if (it["idx"], d) not in done]
    urls = base_urls(args.base_url)
    clients = [OpenAI(base_url=u, api_key=os.environ.get("PRIM_API_KEY", "EMPTY"),
                      timeout=10800.0, max_retries=1) for u in urls]
    print(f"[generation] model={args.model} problems={len(items)} draws={args.draws} "
          f"max_tokens={args.max_tokens} todo={len(todo)} (done {len(done)}) -> {out.path}\n"
          f"[generation] serve the model with --max-model-len 131072 (no --reasoning-parser)", flush=True)

    def run(job):
        k, (it, draw) = job
        t0 = time.time()
        resp = clients[k % len(clients)].chat.completions.create(
            model=args.model, max_completion_tokens=args.max_tokens,
            messages=[{"role": "system", "content": HLE_SYSTEM_PROMPT},
                      {"role": "user", "content": it["question"]}])
        choice = resp.choices[0]
        text = choice.message.content or ""
        return {"idx": it["idx"], "draw": draw, "model": args.model, "question": it["question"],
                "answer": it["answer"], "response": text, "pred": extract_answer_line(text),
                "finish_reason": choice.finish_reason, "usage": usage_dict(resp),
                "elapsed": round(time.time() - t0, 1)}

    n_ok = n_err = 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run, job): job[1] for job in enumerate(todo)}
        for fut in cf.as_completed(futs):
            it, draw = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:  # not written -> retried on the next run
                n_err += 1
                print(f"FAIL {it['idx']} draw={draw}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            if not rec["response"].strip():
                n_err += 1
                print(f"EMPTY {it['idx']} draw={draw}; will retry on rerun", flush=True)
                continue
            out.append(rec)
            n_ok += 1
            if n_ok % 20 == 0 or n_ok + n_err == len(todo):
                print(f"[generation] {n_ok + n_err}/{len(todo)} written={n_ok} failed={n_err}", flush=True)

    recs = out.load()
    print(f"[generation] DONE: {len(recs)}/{len(items) * args.draws} rows on disk, "
          f"{sum(1 for r in recs if r.get('finish_reason') == 'length')} hit the token cap, "
          f"{n_err} failed this run (re-run to retry).", flush=True)


if __name__ == "__main__":
    main()
