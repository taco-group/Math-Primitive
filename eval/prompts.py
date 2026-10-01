"""Prompts for the Prim inference tasks (Discovery, Digestion and Execution).

Do not edit casually: any change to these strings, including the JSON field names in
ExtractedPrimitive (they drive constrained decoding), changes model outputs.
"""
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Discovery: the model writes the primitive for a problem from the question only.
# (str.format placeholder: {question}; literal JSON braces are doubled.)
# ---------------------------------------------------------------------------
DISCOVERY_PROMPT = """You are given a language-only mathematics problem — the problem ONLY, with no solution.

Your task is to produce the single kernel mathematical PRIMITIVE you would use to solve it: the essential idea that makes the problem tractable. Do NOT solve the problem.

Definition:
A mathematical primitive is the essential mathematical property that makes the problem solvable, together with an explanation of how this property connects the problem to a valid solution principle.

The primitive is not merely a theorem name, a technique, or a proof step. For example, "use induction," "apply Riesz representation," or "use generating functions" is insufficient unless it explains what property of the problem makes that method applicable.

Step 1: Identify the essential mathematical property.
Ask: What hidden structure, invariant, relation, decomposition, obstruction, or theorem condition makes the problem tractable?

Step 2: Identify the solution principle.
Ask: How does that property enable a valid theorem, reduction, construction, reformulation, contradiction, or proof strategy? Name the essential conceptual move(s) — one OR several distinct key ideas, whichever the problem genuinely needs.

Step 3: Write the mathematical primitive.
Write 1-3 sentences capturing the single high-level "Aha!" — the central conceptual realization that makes the problem tractable.
CRITICAL CONSTRAINTS:
- Strictly conceptual: no algebraic manipulation, equations, or step-by-step proof.
- HIGH-LEVEL but COMPLETE: state EVERY essential CONCEPTUAL idea the solution rests on (a problem may need ONE or SEVERAL distinct key observations — include each one). Do NOT write out calculations, derivations, or intermediate numeric results.
- Contrast: "recognize the configuration is secretly a Cayley graph" is a mathematical primitive; "build the Cayley graph, compute its spectrum, then sum the eigenvalues to get the count" is a solution outline — produce the former, never the latter.

Step 4: Classify the primitive.
Choose exactly one primary primitive type and zero to three secondary primitive types.
CRITICAL CONSTRAINT: You MUST use the exact string keys provided in the taxonomy below (e.g., "reformulation", "hidden_structure"). Do NOT modify, capitalize, or invent new types.

Do not classify based only on surface-level technique names. Classify based on the essential mathematical property and how it connects to the solution principle.

Primitive taxonomy:

1. reformulation
The problem becomes tractable by rewriting it in a more useful mathematical language, representation, or equivalent form.

2. hidden_structure
The key is recognizing that the objects secretly have a known mathematical structure, such as a group, vector space, Hilbert space, graph, lattice, convex set, metric space, probability space, or algebraic structure.

3. invariant_or_monotonicity
The key is identifying a quantity or property that is preserved, monotone, or constrains all valid transformations.

4. theorem_applicability
The key is recognizing that the problem reduces to verifying the assumptions of a known theorem.

5. duality_or_representation
The key is passing to a dual, transformed, spectral, functional, or representational viewpoint where the problem becomes simpler.

6. extremal_principle
The key is choosing a maximal, minimal, largest, smallest, boundary, or otherwise extreme object and exploiting its extremality.

7. construction_or_witness
The key is explicitly constructing an object, example, counterexample, certificate, function, sequence, graph, or witness.

8. reduction_or_embedding
The key is reducing the problem to another known or canonical problem, or embedding it into a setting where a known result applies.

9. obstruction_or_contradiction
The key is identifying a necessary structural constraint that the opposite assumption would violate.

10. induction_or_recursion
The key is recognizing a recursive, self-similar, or inductive structure and choosing an induction hypothesis that preserves the essential property.

DO NOT SOLVE OR SUMMARIZE THE SOLUTION:
- Do NOT state or compute the final answer; no specific numeric result, final value, or boxed expression.
- Do NOT write a full derivation or proof.
- Do NOT write out the calculations or the step-by-step derivation. State all the essential conceptual primitive(s) the solution rests on — including each distinct key idea it genuinely needs — but not the execution that follows.
Produce the essential mathematical primitive(s): the key thing(s) to notice before starting to solve.

Output ONLY a valid JSON object in the exact format below, with no markdown code blocks formatting (e.g., do not wrap in ```json), just the raw JSON:

{{
  "essential_property": "...",
  "solution_principle": "...",
  "core_concept": "...",
  "primary_primitive_type": "...",
  "secondary_primitive_types": ["..."],
  "classification_rationale": "...",
  "confidence": <integer between 1 and 5>
}}

Problem:
{question}"""

# Appended once if the first attempt leaked the answer (heuristic check in discovery.py).
DISCOVERY_LEAK_REMINDER = '\n\nREMINDER: your previous attempt leaked solution content. Produce ONLY the mathematical primitive — no final answer, no specific numeric result, no boxed expression, and no step-by-step derivation.'

# ---------------------------------------------------------------------------
# Digestion: the model is given the problem AND a correct solution, and names the primitive
# the solution rests on. Identical to DISCOVERY_PROMPT except for the first line, the task
# line, and the appended Solution block. (placeholders: {question} {solution})
# ---------------------------------------------------------------------------
DIGESTION_PROMPT = """You are given a language-only mathematics problem together with a complete, correct solution to it.

Your task is to produce the single kernel mathematical PRIMITIVE that this solution rests on: the essential idea that makes the problem tractable. Do NOT restate or summarize the solution.

Definition:
A mathematical primitive is the essential mathematical property that makes the problem solvable, together with an explanation of how this property connects the problem to a valid solution principle.

The primitive is not merely a theorem name, a technique, or a proof step. For example, "use induction," "apply Riesz representation," or "use generating functions" is insufficient unless it explains what property of the problem makes that method applicable.

Step 1: Identify the essential mathematical property.
Ask: What hidden structure, invariant, relation, decomposition, obstruction, or theorem condition makes the problem tractable?

Step 2: Identify the solution principle.
Ask: How does that property enable a valid theorem, reduction, construction, reformulation, contradiction, or proof strategy? Name the essential conceptual move(s) — one OR several distinct key ideas, whichever the problem genuinely needs.

Step 3: Write the mathematical primitive.
Write 1-3 sentences capturing the single high-level "Aha!" — the central conceptual realization that makes the problem tractable.
CRITICAL CONSTRAINTS:
- Strictly conceptual: no algebraic manipulation, equations, or step-by-step proof.
- HIGH-LEVEL but COMPLETE: state EVERY essential CONCEPTUAL idea the solution rests on (a problem may need ONE or SEVERAL distinct key observations — include each one). Do NOT write out calculations, derivations, or intermediate numeric results.
- Contrast: "recognize the configuration is secretly a Cayley graph" is a mathematical primitive; "build the Cayley graph, compute its spectrum, then sum the eigenvalues to get the count" is a solution outline — produce the former, never the latter.

Step 4: Classify the primitive.
Choose exactly one primary primitive type and zero to three secondary primitive types.
CRITICAL CONSTRAINT: You MUST use the exact string keys provided in the taxonomy below (e.g., "reformulation", "hidden_structure"). Do NOT modify, capitalize, or invent new types.

Do not classify based only on surface-level technique names. Classify based on the essential mathematical property and how it connects to the solution principle.

Primitive taxonomy:

1. reformulation
The problem becomes tractable by rewriting it in a more useful mathematical language, representation, or equivalent form.

2. hidden_structure
The key is recognizing that the objects secretly have a known mathematical structure, such as a group, vector space, Hilbert space, graph, lattice, convex set, metric space, probability space, or algebraic structure.

3. invariant_or_monotonicity
The key is identifying a quantity or property that is preserved, monotone, or constrains all valid transformations.

4. theorem_applicability
The key is recognizing that the problem reduces to verifying the assumptions of a known theorem.

5. duality_or_representation
The key is passing to a dual, transformed, spectral, functional, or representational viewpoint where the problem becomes simpler.

6. extremal_principle
The key is choosing a maximal, minimal, largest, smallest, boundary, or otherwise extreme object and exploiting its extremality.

7. construction_or_witness
The key is explicitly constructing an object, example, counterexample, certificate, function, sequence, graph, or witness.

8. reduction_or_embedding
The key is reducing the problem to another known or canonical problem, or embedding it into a setting where a known result applies.

9. obstruction_or_contradiction
The key is identifying a necessary structural constraint that the opposite assumption would violate.

10. induction_or_recursion
The key is recognizing a recursive, self-similar, or inductive structure and choosing an induction hypothesis that preserves the essential property.

DO NOT SOLVE OR SUMMARIZE THE SOLUTION:
- Do NOT state or compute the final answer; no specific numeric result, final value, or boxed expression.
- Do NOT write a full derivation or proof.
- Do NOT write out the calculations or the step-by-step derivation. State all the essential conceptual primitive(s) the solution rests on — including each distinct key idea it genuinely needs — but not the execution that follows.
Produce the essential mathematical primitive(s): the key thing(s) to notice before starting to solve.

Output ONLY a valid JSON object in the exact format below, with no markdown code blocks formatting (e.g., do not wrap in ```json), just the raw JSON:

{{
  "essential_property": "...",
  "solution_principle": "...",
  "core_concept": "...",
  "primary_primitive_type": "...",
  "secondary_primitive_types": ["..."],
  "classification_rationale": "...",
  "confidence": <integer between 1 and 5>
}}

Problem:
{question}

Solution:
{solution}"""

# 10-type taxonomy used by the structured-output schema.
PrimitiveType = Literal[
    "reformulation",
    "hidden_structure",
    "invariant_or_monotonicity",
    "theorem_applicability",
    "duality_or_representation",
    "extremal_principle",
    "construction_or_witness",
    "reduction_or_embedding",
    "obstruction_or_contradiction",
    "induction_or_recursion",
]


# Structured-output schema for Discovery. The class name, field names and the absence of a
# docstring are part of the protocol: all of them end up in the JSON schema sent to the server.
class ExtractedPrimitive(BaseModel):
    essential_property: str
    solution_principle: str
    core_concept: str
    primary_primitive_type: PrimitiveType
    secondary_primitive_types: list[PrimitiveType] = Field(default_factory=list, max_length=3)
    classification_rationale: str
    confidence: int = Field(ge=1, le=5)


# ---------------------------------------------------------------------------
# Execution: the model is handed the gold primitive and solves the problem.
# (placeholders: {problem} {essential_property} {solution_principle} {core_concept})
# ---------------------------------------------------------------------------
EXECUTION_PROMPT = """You are a math problem solver. Below is a math problem and a high-quality mathematical primitive that has been independently validated as capturing the right core structure for this problem. Use the primitive to guide your derivation, but you must still derive each step rigorously and produce the final answer in the form the problem requires.

## Problem
{problem}

## Validated Primitive
essential_property: {essential_property}

solution_principle: {solution_principle}

core_concept: {core_concept}

## Output requirements
Write a careful derivation that follows the primitive. End with a `## Final Answer` block containing ONLY the final answer in the literal form the problem asks for. No extra commentary after the Final Answer line."""
