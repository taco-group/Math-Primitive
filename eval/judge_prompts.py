"""Judge prompts and schemas.

Generation / Execution: the official HLE judge (centerforaisafety/hle, hle_eval/run_judge_results.py),
verbatim, with its default judge model o3-mini-2025-01-31.

Discovery / Digestion: the primitive judge. Three independent structured calls per problem:
validity V (is the output a primitive at all), GATE (the essential WHAT) and MECHANISM (the HOW).
    Score = V * sigma_gate * (0.6 + 0.4 * sigma_mech),   PrimitiveAcc = 1[Score >= 0.8]
"""
from typing import Literal

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# HLE official judge (verbatim)
# ---------------------------------------------------------------------------
HLE_JUDGE_MODEL = "o3-mini-2025-01-31"

HLE_JUDGE_PROMPT = "Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.\n\n[question]: {question}\n\n[response]: {response}\n\nYour judgement must be in the format and criteria specified below:\n\nextracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.\n\n[correct_answer]: {correct_answer}\n\nreasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.\n\ncorrect: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.\n\n\nconfidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available."


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True]  # 100% reliability


# ---------------------------------------------------------------------------
# Primitive judge
# ---------------------------------------------------------------------------
PRIMITIVE_JUDGE_MODEL = "gpt-5.4"
PRIMITIVE_JUDGE_EFFORT = "medium"
TAU = 0.8

VALIDITY_PROMPT = """You are checking whether a candidate mathematical "primitive" for a mathematics problem is VACUOUS — i.e. not really a primitive at all. Look only at the candidate below.

A mathematical primitive names the essential property that makes the problem solvable and how it connects to a solution.

Mark verdict = "invalid" ONLY if the candidate is VACUOUS, i.e. ANY of:
- "generic": vague advice with no problem-specific content ("use a clever substitution", "find the pattern").
- "technique_name_only": names a method/theorem but never states the property of THIS problem that makes it apply.
- "computation_only": only arithmetic/numeric work, with no conceptual claim.
Otherwise verdict = "valid", failure = "none".

Do NOT mark invalid for stating a conclusion or a final result, and do NOT mark invalid for being a full derivation — over-completeness is NOT vacuity. A thorough or conclusion-stating answer can still contain a valid primitive. (Answer leakage is handled separately, not here.)

CANDIDATE PRIMITIVE:
  essential_property: {essential_property}
  solution_principle: {solution_principle}
  core_concept: {core_concept}

Return JSON only: {{"verdict": "valid"|"invalid", "failure": "none"|"generic"|"technique_name_only"|"computation_only", "evidence": "<one short sentence>"}}"""

GATE_DEF = "the GATE = the HIGH-LEVEL conceptual idea (the 'aha') the solution is built on — the single problem-specific reframing / structure / property that unlocks it. The gold primitive's essential_property pinpoints it; the reference solution shows why it is essential. This is the IDEA, not the execution that follows from it."
MECH_DEF = 'the MECHANISM = the HIGH-LEVEL reason that idea unlocks the problem (which known result / reduction / construction / argument it conceptually enables) — NOT the detailed derivation.'
GATE_CRIT = {'match': "the model states the correct problem-specific CORE high-level idea that unlocks the problem (the dominant 'aha' the solution is built on — the same as the reference, or an equivalent / valid-alternative one), even if coarser or differently worded. Do NOT lower this for missing anything DOWNSTREAM of that idea: secondary lemmas, second-step ideas, lower-bound / sharpness / optimality arguments, exact constants, specific numeric values, or calculation steps — those are the execution the core primitive enables, not the primitive.", 'partial': 'the CORE high-level idea itself is vague/generic or only half-right (the central aha is incomplete or distorted) — NOT merely that some downstream idea / bound / computation is missing.', 'mismatch': 'the core high-level idea is wrong, peripheral, or a different idea that would not unlock the problem.'}
MECH_CRIT = {'match': "the model correctly connects the idea to a solution at the CONCEPTUAL level (the reference's high-level route, or a valid equivalent), even if it omits the detailed derivation.", 'partial': 'the high-level connection is only partly right or left unclear (not merely missing execution detail).', 'mismatch': 'the connection is wrong or missing.'}

SLOT_PROMPT = """You are judging whether a MODEL's candidate mathematical primitive for a math problem is CORRECT on ONE dimension, using a known-correct REFERENCE SOLUTION. Judge correctness FOR THIS PROBLEM — not reproduction of any particular wording.

GROUND RULES (read carefully):
- The REFERENCE SOLUTION below is correct and complete. The GOLD PRIMITIVE is a terse distillation of its key idea, written WITH the full solution in hand. The MODEL only saw the problem.
- Judge whether the MODEL's primitive correctly captures the {dim} that unlocks this problem, as evidenced by the reference solution.
- CREDIT a correct primitive even if it is (i) phrased more coarsely or less precisely than the gold primitive, or (ii) a different but valid route that the reference solution shows would work.
- Do NOT require the model to reproduce the gold primitive's exact framing, or to name the same theorem if its statement is equivalent.
- Do NOT re-solve from scratch; judge the model's primitive against the reference solution.
- Do NOT consider whether the model reached any final answer — judge the primitive content only.
- A primitive here is the SINGLE CORE high-level idea (the dominant 'aha'). Everything DOWNSTREAM of it is execution, NOT part of the primitive, and its absence must NOT lower the score: secondary lemmas, second-step ideas, lower-bound / sharpness / optimality arguments, exact constants, specific numeric values, and calculation steps are what the core primitive ENABLES. Judge only whether the model has the correct CORE idea. (A vague, non-problem-specific statement, however, is still not a primitive.)

The dimension you are judging is the **{dim}**:
{dim_def}

Scoring (CORRECTNESS, not reproduction):
- "match": {c_match}
- "partial": {c_partial}
- "mismatch": {c_mismatch}

PROBLEM:
{question}

REFERENCE SOLUTION (correct):
{rationale}

GOLD PRIMITIVE (terse pointer to the essential idea):
  essential_property: {g_ep}
  solution_principle: {g_sp}
  core_concept: {g_ci}

MODEL PRIMITIVE (judge this):
  essential_property: {m_ep}
  solution_principle: {m_sp}
  core_concept: {m_ci}

Return JSON only: {{"agreement": "match"|"partial"|"mismatch", "evidence": "<one short sentence>"}}"""


class ValidityJudgment(BaseModel):
    verdict: Literal["valid", "invalid"]
    failure: Literal["none", "generic", "technique_name_only", "computation_only"]
    evidence: str


class SlotJudgment(BaseModel):
    agreement: Literal["match", "partial", "mismatch"]
    evidence: str


SIGMA = {"match": 1.0, "partial": 0.5, "mismatch": 0.0}


def slot_prompt(dim: str, question: str, solution: str, gold: dict, model: dict) -> str:
    """dim = "gate" or "mech". Question / solution are truncated as in the paper runs."""
    if dim == "gate":
        label, dim_def, crit = "GATE (the essential WHAT)", GATE_DEF, GATE_CRIT
    else:
        label, dim_def, crit = "MECHANISM (the HOW)", MECH_DEF, MECH_CRIT
    return SLOT_PROMPT.format(
        dim=label, dim_def=dim_def,
        c_match=crit["match"], c_partial=crit["partial"], c_mismatch=crit["mismatch"],
        question=(question or "")[:4000], rationale=(solution or "")[:6000],
        g_ep=gold.get("essential_property", ""), g_sp=gold.get("solution_principle", ""),
        g_ci=gold.get("core_concept", ""),
        m_ep=model.get("essential_property", ""), m_sp=model.get("solution_principle", ""),
        m_ci=model.get("core_concept", ""),
    )
