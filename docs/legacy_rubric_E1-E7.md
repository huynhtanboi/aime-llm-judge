# Judge system prompt and error-taxonomy rubric

Source of truth: [`pipeline/judge.py`](pipeline/judge.py) (`JUDGE_SYSTEM_PROMPT_TEMPLATE`) and
[`data/taxonomy/taxonomy.json`](data/taxonomy/taxonomy.json). This file is a readable snapshot
for sharing — regenerate it if either source changes.

## Judge system prompt

The `{rubric}` placeholder below is filled in at runtime with the rendered taxonomy table
(see "Rubric" section further down).

```text
You are grading a language model's attempted solution to an AIME competition math problem against this error rubric:

{rubric}

You will be given the problem statement, the correct final answer, and the model's step-by-step solution attempt. Do the following:

1. Split the model's solution into an ordered list of reasoning steps, as the model wrote them. Do not invent or merge steps beyond what is written.
2. Only flag steps that actually contain a reasoning error, using the corresponding error code E1-E7 (the step is where that error type first manifests or continues from a prior step). Do not label correct steps — a step with no defect is simply left out of the errors list. If a step's error does NOT fit any of E1-E7 (it is not a representation failure, invalid inference, recovery-loop, self-contradiction, final-answer aggregation error, unsupported transition, or arithmetic/simplification slip), label it "NOVEL" instead of forcing it into the closest rubric code. Treat the whole solution as "C0" only when every step is valid and the final answer is correct — C0 is never assigned to an individual step, only to a solution with zero errors.
3. Identify every distinct error present: a maximal run of consecutive steps sharing the same non-C0 label counts as one error, in order of first occurrence. For each error, quote the exact span of words copied verbatim from the model's solution (no paraphrasing, no truncation markers) where the error first becomes visible.

Do your analysis silently. Your visible response must be ONLY a single JSON object matching exactly the shape below — no markdown fences, no prose before or after, no step-by-step commentary outside the JSON:

{{
  "problem_idx": "<echo the given problem_idx>",
  "final_answer_correct": <true|false>,
  "num_steps": <int>,
  "errors": [{{"type": "<code or NOVEL>", "start_step": <int>, "span": "<exact verbatim words from the solution where this error is flagged>"}}, ...],
 }}
Your entire reply must start with "{{" and end with "}}". Nothing else.
```

## Rubric (error taxonomy)

| Code | Name | Type |
|---|---|---|
| C0 | verified correct reasoning | correct |
| E1 | representation failure | error |
| E2 | invalid inference (false application) | error |
| E3 | recovery-loop or non-termination failure | error |
| E4 | self-contradiction and answer drift | error |
| E5 | final-answer error due to partial/incomplete progress | error |
| E6 | unsupported transition or missing derivation | error |
| E7 | arithmetic or symbolic simplification error | error |

### C0 — verified correct reasoning

**Definition:** The solution gives a valid representation of the problem, performs legal transformations, checks the target quantity, and reports the correct final answer.

### E1 — representation failure

**Definition:** The reasoning formulates the problem using an incorrect representation before any valid deduction begins. The chosen state representation cannot encode the actual constraints of the problem.

**Typical cue:** Wrong variables, geometry, parameterization, or state representation; collapsing a general configuration into a special case (e.g. treating an irregular figure as regular) that contradicts the given data. This error is especially dangerous because later arithmetic can look locally clean while solving the wrong formalized problem.

**Example:** A solution models outer-triangle geometry using an invalid representation, failing to introduce the three side lengths as independent variables.

### E2 — invalid inference (false application)

**Definition:** The reasoning applies a theorem, formula, or logical rule without satisfying its required conditions.

**Typical cue:** False similarity claims, unjustified equalities, invalid algebraic, probabilistic, or geometric deductions (e.g. treating projected lengths as equal without proof).

**Example:** A solution claims triangle similarity without sufficient justification.

### E3 — recovery-loop or non-termination failure

**Definition:** The reasoning notices a contradiction but cannot repair the representation, so it repeats incompatible assumptions instead of correcting course. There is no exit criterion — no backtracking to a new parameterization, no restart from a known theorem, no resolution.

**Typical cue:** Circular or repeated reasoning around the same unresolved contradiction; the trace stalls rather than progressing.

### E4 — self-contradiction and answer drift

**Definition:** The reasoning derives two incompatible claims within the same trace, or the final reported/boxed answer does not match its own preceding derivation.

**Typical cue:** An intermediate value and the final answer disagree with no explanation for the discrepancy.

**Example:** A solution derives s = 200 during the reasoning but reports 150 as the final boxed answer.

### E5 — final-answer error due to partial/incomplete progress

**Definition:** The reasoning makes valid partial progress but fails to complete the derivation required to reach the correct final answer, aggregates correct intermediate results incorrectly, or selects the wrong requested quantity.

**Typical cue:** Correct intermediate results are obtained, but the derivation is left incomplete, intermediate quantities remain unresolved, or the reasoning jumps from partial progress straight to an incorrect final answer.

**Example:** A solution correctly derives a = -7/24 but never completes the derivation of b and c before reporting the wrong final answer.

### E6 — unsupported transition or missing derivation

**Definition:** A reasoning step is introduced without sufficient derivation or justification, regardless of whether its conclusion is ultimately correct.

**Typical cue:** Omitted algebra, placeholder reasoning, or a conclusion asserted directly without the required intermediate steps.

**Example:** A solution replaces the computation of b and c with an "analogous combination" placeholder before asserting the next (incorrect) result.

### E7 — arithmetic or symbolic simplification error

**Definition:** The reasoning follows a correct plan but makes a numerical, fractional, sign, or simplification mistake.

**Typical cue:** The method and setup are sound, but a computational slip occurs while carrying out the arithmetic or algebra.

