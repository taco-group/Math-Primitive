"""Primitive judge for Discovery and Digestion outputs.

For each problem the model's primitive is compared with the expert-verified gold primitive,
with the reference solution as supporting context, by three independent structured calls:
    V           in {0, 1}        validity: is the output a primitive at all (not vacuous)
    sigma_gate  in {0, .5, 1}    does it identify the essential structure (the WHAT)
    sigma_mech  in {0, .5, 1}    does it explain how that structure enables the solution (the HOW)
    Score = V * sigma_gate * (0.6 + 0.4 * sigma_mech);  a primitive counts as found if Score >= 0.8.
The same judge scores Discovery and Digestion. Each row also carries the problem's primitive
family (Recast / Witness / Argument) for the per-family breakdown in score.py.

This calls the OpenAI API and costs money (3 calls per problem): it needs OPENAI_API_KEY.

Example:
    python eval/judge_primitive.py --input results/qwen-3.5-9b-absorb/discovery.jsonl
    python eval/judge_primitive.py --input results/qwen-3.5-9b-absorb/digestion.jsonl
Writes <input>.judged.jsonl next to the input and resumes from it if it exists.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import JsonlWriter, load_prim  # noqa: E402
from judge_prompts import (PRIMITIVE_JUDGE_EFFORT, PRIMITIVE_JUDGE_MODEL, SIGMA, VALIDITY_PROMPT,  # noqa: E402
                           SlotJudgment, ValidityJudgment, slot_prompt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="discovery.jsonl or digestion.jsonl")
    ap.add_argument("--output", default=None, help="default: <input>.judged.jsonl")
    ap.add_argument("--judge-model", default=PRIMITIVE_JUDGE_MODEL)
    ap.add_argument("--reasoning-effort", default=PRIMITIVE_JUDGE_EFFORT)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set: the primitive judge runs on the OpenAI API.")

    prim = {it["idx"]: it for it in load_prim()}
    rows = [r for r in JsonlWriter(args.input).load() if r.get("prediction")]  # failed rows score 0
    out = JsonlWriter(args.output or str(Path(args.input).with_suffix("")) + ".judged.jsonl")
    done = {r["idx"]: r for r in out.load() if r.get("score") is not None}
    todo = [r for r in rows if r["idx"] not in done]
    client = OpenAI(timeout=3600.0, max_retries=1)
    print(f"[judge_primitive] {args.input}: rows with a prediction={len(rows)} already judged={len(done)} "
          f"todo={len(todo)} judge={args.judge_model}@{args.reasoning_effort} -> {out.path}", flush=True)

    def call(prompt, schema):
        kwargs = dict(model=args.judge_model, messages=[{"role": "user", "content": prompt}],
                      response_format=schema, max_completion_tokens=args.max_tokens)
        if args.reasoning_effort:
            kwargs["reasoning_effort"] = args.reasoning_effort
        for _ in range(3):
            try:
                return client.chat.completions.parse(**kwargs).choices[0].message.parsed, None
            except Exception as e:
                msg = str(e)
                if "reasoning_effort" in msg and "reasoning_effort" in kwargs:
                    kwargs.pop("reasoning_effort"); continue
                if "max_completion_tokens" in msg and "max_completion_tokens" in kwargs:
                    kwargs.pop("max_completion_tokens"); continue
                return None, f"{type(e).__name__}: {e}"
        return None, "retries exhausted"

    def run(r: dict) -> dict:
        it = prim[r["idx"]]
        gold, pred = it["primitive"], r["prediction"]
        prompts = [
            (VALIDITY_PROMPT.format(essential_property=pred.get("essential_property", ""),
                                    solution_principle=pred.get("solution_principle", ""),
                                    core_concept=pred.get("core_concept", "")), ValidityJudgment),
            (slot_prompt("gate", it["question"], it["solution"], gold, pred), SlotJudgment),
            (slot_prompt("mech", it["question"], it["solution"], gold, pred), SlotJudgment),
        ]
        with cf.ThreadPoolExecutor(3) as ex:
            (v, ve), (g, ge), (m, me) = list(ex.map(lambda a: call(*a), prompts))
        rec = {"idx": r["idx"], "model": r.get("model"), "family": it["family"], "score": None}
        if v is None or g is None or m is None:
            rec["error"] = ve or ge or me or "judge returned no parsed object"
            return rec
        V = 1.0 if v.verdict == "valid" else 0.0
        sg, sm = SIGMA[g.agreement], SIGMA[m.agreement]
        rec.update(V=V, sigma_gate=sg, sigma_mech=sm, score=V * sg * (0.6 + 0.4 * sm),
                   validity=v.model_dump(), gate=g.model_dump(), mech=m.model_dump(),
                   judge_model=args.judge_model, error=None)
        return rec

    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run, r) for r in todo]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            rec = fut.result()
            if rec["score"] is not None:
                done[rec["idx"]] = rec
            else:
                print(f"FAIL {rec['idx']}: {rec['error'][:200]}", flush=True)
            if i % 10 == 0 or i == len(futs):
                out.rewrite([done[x["idx"]] for x in rows if x["idx"] in done])
                print(f"[judge_primitive] {i}/{len(futs)}", flush=True)

    judged = [done[x["idx"]] for x in rows if x["idx"] in done]
    out.rewrite(judged)
    print(f"[judge_primitive] DONE judged={len(judged)}/{len(rows)} "
          f"score>=0.8: {sum(1 for r in judged if r['score'] >= 0.8)}. Re-run to judge rows that failed.",
          flush=True)


if __name__ == "__main__":
    main()
