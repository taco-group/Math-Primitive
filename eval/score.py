"""Score one model on Prim from the judged files (no API calls).

    python eval/score.py --results results/qwen-3.5-9b-absorb

Reads whichever of these exist in the results directory:
    generation.judged.jsonl  execution.judged.jsonl   (from judge_answer.py)
    discovery.judged.jsonl   digestion.judged.jsonl   (from judge_primitive.py)
and reports, as in the paper, over all 182 Prim problems (a missing, failed or unjudged row
counts as wrong):
    Generation, Execution   = % judged correct by the HLE judge
    Discovery,  Digestion   = PrimitiveAcc = % with primitive Score >= 0.8
plus Digestion - Discovery and Execution - Generation, and a per-family breakdown.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import JsonlWriter, load_prim  # noqa: E402
from judge_prompts import TAU  # noqa: E402

FAMILIES = ("Recast", "Witness", "Argument")


def load(path: Path) -> list[dict]:
    return JsonlWriter(path).load() if path.exists() else []


def answer_correct(rows: list[dict]) -> dict[str, float]:
    """idx -> fraction of draws judged correct (1 draw in the paper protocol)."""
    by: dict[str, list[bool]] = {}
    for r in rows:
        by.setdefault(r["idx"], []).append((r.get("judge") or {}).get("correct") == "yes")
    return {k: sum(v) / len(v) for k, v in by.items()}


def primitive_correct(rows: list[dict], tau: float) -> dict[str, float]:
    return {r["idx"]: float((r.get("score") or 0.0) >= tau) for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results/<served-name> directory")
    ap.add_argument("--tau", type=float, default=TAU)
    args = ap.parse_args()
    d = Path(args.results)
    items = load_prim()
    fam = {it["idx"]: it["family"] for it in items}

    dims = {
        "Discovery": primitive_correct(load(d / "discovery.judged.jsonl"), args.tau),
        "Generation": answer_correct(load(d / "generation.judged.jsonl")),
        "Digestion": primitive_correct(load(d / "digestion.judged.jsonl"), args.tau),
        "Execution": answer_correct(load(d / "execution.judged.jsonl")),
    }
    present = {k for k in dims if (d / f"{k.lower()}.judged.jsonl").exists()}

    def acc(scores: dict[str, float], ids: list[str]) -> float:
        return 100.0 * sum(scores.get(i, 0.0) for i in ids) / len(ids)

    ids = [it["idx"] for it in items]
    res = {k: acc(v, ids) for k, v in dims.items() if k in present}
    print(f"Prim results for {d.name} (n = {len(ids)}; missing rows count as wrong; tau = {args.tau})")
    for k in ("Discovery", "Generation", "Digestion", "Execution"):
        if k not in res:
            print(f"  {k:<11} -  (no {k.lower()}.judged.jsonl)")
            continue
        extra = ""
        if k == "Digestion" and "Discovery" in res:
            extra = f"   Dig. - Disc. = {res['Digestion'] - res['Discovery']:+.2f}"
        if k == "Execution" and "Generation" in res:
            extra = f"   Exec. - Gen. = {res['Execution'] - res['Generation']:+.2f}"
        print(f"  {k:<11} {res[k]:6.2f}{extra}")

    print("\nBy primitive family (" + " / ".join(f"{f} n={sum(1 for i in ids if fam[i] == f)}" for f in FAMILIES) + ")")
    for k in ("Discovery", "Generation", "Digestion", "Execution"):
        if k in res:
            cells = [f"{acc(dims[k], [i for i in ids if fam[i] == f]):6.2f}" for f in FAMILIES]
            print(f"  {k:<11} " + "  ".join(cells))


if __name__ == "__main__":
    main()
