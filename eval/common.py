"""Shared helpers for the Prim inference scripts.

Prim is loaded from the Hugging Face Hub (shuoxing/Prim). Every row is normalised to
    {"idx": "prim-0000", "question": str, "answer": str, "solution": str, "primitive": dict,
     "family": "Recast" | "Witness" | "Argument"}
with primitive = {"essential_property", "solution_principle", "core_concept"}.

Shared by the inference scripts and the judges.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

PRIM_REPO = os.environ.get("PRIM_DATASET", "shuoxing/Prim")

# HLE's official answer-format system prompt, verbatim from
# centerforaisafety/hle/hle_eval/run_model_predictions.py (Prim problems come from HLE).
HLE_SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)


def load_prim() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(PRIM_REPO, split="test")
    return [{"idx": f"prim-{i:04d}", "question": r["question"], "answer": r["answer"],
             "solution": r["solution"], "primitive": r["primitive"], "family": r["family"]}
            for i, r in enumerate(ds)]


# ---------------------------------------------------------------------------
# Answer extraction (for convenience only; judges do their own extraction)
# ---------------------------------------------------------------------------
def _balanced(text: str, open_idx: int) -> str | None:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i]
    return None


def extract_answer_line(text: str | None) -> str | None:
    """HLE format: last `Answer:` line, else the last \\boxed{...}."""
    if not text:
        return None
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s.lower().startswith("answer:"):
            return s.split(":", 1)[1].strip()
    last = None
    for m in re.finditer(r"\\boxed\s*\{", text):
        inner = _balanced(text, m.end() - 1)
        if inner is not None:
            last = inner
    return last.strip() if last else None


def extract_final_answer_block(text: str | None) -> str | None:
    """Execution prompt format: the `## Final Answer` block."""
    if not text:
        return None
    m = re.search(r"##\s*Final\s+Answer\s*\n([\s\S]*?)(?=\n\s*##|\s*\Z)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# JSONL output with resume
# ---------------------------------------------------------------------------
class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    def append(self, rec: dict) -> None:
        with self._lock, open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def rewrite(self, recs: list[dict]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with self._lock, open(tmp, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)


def base_urls(arg: str | None) -> list[str]:
    """Comma-separated OpenAI-compatible base URLs (e.g. http://localhost:8000/v1)."""
    raw = arg or os.environ.get("PRIM_BASE_URL", "http://localhost:8000/v1")
    return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]


def usage_dict(resp) -> dict:
    u = getattr(resp, "usage", None)
    return u.model_dump() if u is not None else {}
