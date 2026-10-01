"""Digestion on Prim: the model is given the problem AND a complete correct solution, and names
the primitive that solution rests on (pi(x, y) -> p).

Protocol (as in the paper):
  * prompt = prompts.DIGESTION_PROMPT (the Discovery prompt with the solution added), single
    user message, no system prompt
  * structured output (JSON schema of prompts.ExtractedPrimitive, constrained decoding on vLLM)
  * reasoning_effort="high", max_completion_tokens=32768; serve with --max-model-len 40960
  * ONE attempt: unlike Discovery there is no retry. The solution is in context, so the
    answer-leak check would fire on a non-random subset and turn a retry into best-of-2.
    `leaked` is still recorded.

The output primitive is scored afterwards against the gold primitive by eval/judge_primitive.py.

Example:
    python eval/digestion.py --model qwen-3.5-9b-absorb --base-url http://localhost:8000/v1
Re-run the same command to retry rows that failed.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import JsonlWriter, base_urls, load_prim, usage_dict  # noqa: E402
from discovery import leakage_check  # noqa: E402
from prompts import DIGESTION_PROMPT, ExtractedPrimitive  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible base URL(s), comma-separated")
    ap.add_argument("--out", default=None, help="default: results/<model>/digestion.jsonl")
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    items = load_prim()[: args.limit]
    out = JsonlWriter(args.out or f"results/{args.model}/digestion.jsonl")
    prev = {r["idx"]: r for r in out.load()}
    todo = [it for it in items if not (prev.get(it["idx"]) or {}).get("prediction")]
    urls = base_urls(args.base_url)
    clients = [OpenAI(base_url=u, api_key=os.environ.get("PRIM_API_KEY", "EMPTY"),
                      timeout=3600.0, max_retries=1) for u in urls]
    print(f"[digestion] model={args.model} rows={len(items)} todo={len(todo)} -> {out.path}\n"
          f"[digestion] serve the model with --max-model-len 40960 (no --reasoning-parser)", flush=True)

    def call(client, prompt):
        kwargs = dict(model=args.model, messages=[{"role": "user", "content": prompt}],
                      response_format=ExtractedPrimitive, max_completion_tokens=args.max_tokens)
        if args.reasoning_effort:
            kwargs["reasoning_effort"] = args.reasoning_effort
        for _ in range(3):  # drop parameters a server rejects, then retry
            try:
                return client.chat.completions.parse(**kwargs)
            except Exception as e:
                msg = str(e)
                if "reasoning_effort" in msg and "reasoning_effort" in kwargs:
                    kwargs.pop("reasoning_effort"); continue
                if "max_completion_tokens" in msg and "max_completion_tokens" in kwargs:
                    kwargs.pop("max_completion_tokens"); continue
                raise
        return None

    def run(job):
        k, it = job
        t0 = time.time()
        rec = {"idx": it["idx"], "model": args.model, "question": it["question"], "prediction": None,
               "leaked": False, "leak_reason": "", "attempts": 1, "raw_response": "", "error": None, "usage": {}}
        try:
            resp = call(clients[k % len(clients)],
                        DIGESTION_PROMPT.format(question=it["question"], solution=it["solution"]))
            msg = resp.choices[0].message if resp else None
            rec["raw_response"] = (getattr(msg, "content", "") or "") if msg else ""
            rec["usage"] = usage_dict(resp) if resp else {}
            parsed = getattr(msg, "parsed", None) if msg else None
            if parsed is None:
                rec["error"] = "structured-output parse returned None"
            else:
                rec["prediction"] = parsed.model_dump()
                rec["leaked"], rec["leak_reason"] = leakage_check(rec["prediction"], it["answer"])
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["elapsed"] = round(time.time() - t0, 1)
        return rec

    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run, job) for job in enumerate(todo)]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            rec = fut.result()  # run() never raises; errors are recorded in the row
            prev[rec["idx"]] = rec
            if rec["error"]:
                print(f"FAIL {rec['idx']}: {rec['error'][:200]}", flush=True)
            if i % 10 == 0 or i == len(todo):
                out.rewrite([prev[it["idx"]] for it in items if it["idx"] in prev])
                print(f"[digestion] {i}/{len(todo)}", flush=True)

    recs = [prev[it["idx"]] for it in items if it["idx"] in prev]
    out.rewrite(recs)
    ok = sum(1 for r in recs if r.get("prediction"))
    print(f"[digestion] DONE ok={ok}/{len(items)} failed={len(items) - ok} "
          f"leaked(flagged)={sum(1 for r in recs if r.get('leaked'))}. Re-run to retry failed rows.", flush=True)


if __name__ == "__main__":
    main()
