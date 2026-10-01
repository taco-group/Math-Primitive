"""Answer judge for Generation and Execution outputs: the official HLE judge.

Each response is compared with the gold `answer` by the HLE judge prompt (verbatim from
centerforaisafety/hle) with structured output; the judge extracts the final answer and returns
correct = "yes" | "no". Empty responses are marked incorrect without a judge call.

This calls the OpenAI API and costs money: it needs OPENAI_API_KEY in the environment.

Example:
    python eval/judge_answer.py --input results/qwen-3.5-9b-absorb/generation.jsonl
    python eval/judge_answer.py --input results/qwen-3.5-9b-absorb/execution.jsonl
Writes <input>.judged.jsonl next to the input and resumes from it if it exists.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import JsonlWriter  # noqa: E402
from judge_prompts import HLE_JUDGE_MODEL, HLE_JUDGE_PROMPT, ExtractedAnswer  # noqa: E402


def key(r: dict) -> tuple:
    return (r["idx"], r.get("draw", 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="generation.jsonl or execution.jsonl")
    ap.add_argument("--output", default=None, help="default: <input>.judged.jsonl")
    ap.add_argument("--judge-model", default=HLE_JUDGE_MODEL)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=100)
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set: the HLE judge runs on the OpenAI API.")

    rows = JsonlWriter(args.input).load()
    out = JsonlWriter(args.output or str(Path(args.input).with_suffix("")) + ".judged.jsonl")
    done = {key(r): r for r in out.load() if r.get("judge")}
    todo = [r for r in rows if key(r) not in done]
    client = OpenAI(timeout=300.0, max_retries=2)
    print(f"[judge_answer] {args.input}: rows={len(rows)} already judged={len(done)} todo={len(todo)} "
          f"judge={args.judge_model} -> {out.path}", flush=True)

    def run(r: dict) -> dict:
        r = dict(r)
        response = r.get("response") or ""
        if not response.strip():
            r["judge"] = {"model_answer": "None", "reasoning": "No response.", "correct": "no",
                          "confidence": 100, "judge_model": args.judge_model}
            return r
        prompt = HLE_JUDGE_PROMPT.format(question=r["question"], correct_answer=r["answer"], response=response)
        for attempt in range(3):
            try:
                resp = client.chat.completions.parse(
                    model=args.judge_model, max_completion_tokens=args.max_tokens,
                    messages=[{"role": "user", "content": prompt}], response_format=ExtractedAnswer)
                p = resp.choices[0].message.parsed
                if p is None:
                    raise RuntimeError("judge returned no parsed object")
                r["judge"] = {"model_answer": p.extracted_final_answer, "reasoning": p.reasoning,
                              "correct": p.correct, "confidence": p.confidence, "judge_model": args.judge_model}
                return r
            except Exception as e:  # retried with backoff; left unjudged after 3 failures
                r["judge_error"] = f"{type(e).__name__}: {e}"
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return r

    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run, r) for r in todo]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = fut.result()
            if r.get("judge"):
                done[key(r)] = r
            else:
                print(f"FAIL {key(r)}: {r.get('judge_error', '')[:200]}", flush=True)
            if i % 25 == 0 or i == len(futs):
                out.rewrite([done[key(x)] for x in rows if key(x) in done])
                print(f"[judge_answer] {i}/{len(futs)}", flush=True)

    judged = [done[key(x)] for x in rows if key(x) in done]
    out.rewrite(judged)
    ok = sum(1 for r in judged if r["judge"]["correct"] == "yes")
    print(f"[judge_answer] DONE judged={len(judged)}/{len(rows)} correct={ok}. "
          f"Re-run to judge rows that failed.", flush=True)


if __name__ == "__main__":
    main()
