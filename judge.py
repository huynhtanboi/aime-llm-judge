"""LLM judge for AIME solutions: step-level error type / span annotation.

Given (problem, solver's solution, ground-truth answer), an LLM judge splits the
solution into steps and returns every independent reasoning error with
  - type        : E1-E6 or NOVEL (see JUDGE_SYSTEM_PROMPT; no errors == C0)
  - start_step  : 1-based step where the error first becomes visible
  - span        : shortest verbatim substring of the solution showing the error

The module also contains the data loader used to read problems/ground truth,
solver outputs and judge outputs, and to join them into flat records.

Usage:
  python judge.py run    --solutions data/solutions/greedy/qwen2.5_7b_aime2024.json
  python judge.py export --solutions data/solutions/greedy/qwen2.5_7b_aime2024.json \
                         --judgments judge_outputs/greedy/gpt-5.6-luna_judge_qwen2.5_7b_aime2024.json
"""
import argparse
import glob
import json
import os
import re

# ---------------------------------------------------------------------------
# Judge prompt / rubric
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = r"""# AIME Solution Judge Prompt and Rubric

You are grading a language model's attempted solution to an AIME competition math problem, particularly identifying its error types and locations.

You will be given:

- the problem statement,

- the correct final answer, and

- the model's solution attempt.

Judge the mathematical reasoning independently. Do not assume that a step is valid merely because it resembles a standard method or because the final answer is correct.

## Required procedure

### 1. Split the solution into steps

Create an ordered list of the steps actually present in the model's response.

Use the following deterministic rules:

1. If the response contains explicit numbered steps or headings such as `Step 1`, use each complete numbered block as one step.

2. Otherwise, treat each top-level prose paragraph or bullet item as one step.

3. Keep displayed equations with the prose immediately introducing or explaining them. Consecutive equations belonging to one calculation remain in the same step.

4. A standalone final-answer line is a separate step only when it is visibly separate from the preceding paragraph.

5. Do not invent, merge, or subdivide steps based on how you would prefer the solution to be written.

### 2. Determine final-answer correctness (for your own reasoning only; not part of the output)

Internally determine whether the final answer is correct by comparing the model's last unambiguous proposed answer with the supplied correct answer. This determination is not reported as a field -- it only informs how you apply the checks below and the E4 category.

Even when the final answer is correct, the reasoning trace may still contain one or more errors. A correct answer may result from guessing, cancellation of mistakes, an unsupported shortcut, or invalid reasoning that happens to reach the expected value. Therefore, evaluate every reasoning step independently of final-answer correctness and report all trace-level errors that satisfy the rubric.

- Equivalent exact forms count as equal.

- An approximation counts only when it uniquely and correctly determines the required AIME integer.

- If the response gives multiple incompatible final answers, use the last unambiguous answer for this internal determination and consider whether E4 applies.

- A wrong final answer is an outcome, not automatically a separate reasoning error. Do not flag the final line merely because it disagrees with the reference answer.

### 3. Identify independent root errors

Flag each independent error at the earliest step where it becomes visible.

- Prefer the root cause over downstream consequences.

- Do not repeatedly flag later calculations that merely propagate an already identified error.

- If a later step introduces a new independent defect, flag that new defect.

- A maximal consecutive run caused by the same defect receives one error entry, located at the first step in that run.

- Do not assign two error codes to the same span.

- Multiple errors may share a step number only when they are genuinely independent and have distinct, nonoverlapping spans.

- Correct but concise reasoning is not an error. Do not demand unnecessary expansion of routine algebra or standard facts.

- Even when the final answer is correct, the reasoning trace may still contain one or more errors. A correct answer may result from guessing, cancellation of mistakes, an unsupported shortcut, or invalid reasoning that happens to reach the expected value.

- A wrong final answer is therefore an outcome, not automatically a separate reasoning error. Do not flag the final line merely because it disagrees with the reference answer.

### 4. Select exactly one error code

Allowed error codes are:

- `E1`,

- `E2`,

- `E3`,

- `E4`,

- `E5`,

- `E6`,

- `NOVEL`.

Use the following decision order below when two codes appear plausible:

1. E1 for a fundamentally wrong initial representation.

2. E6 for a localized arithmetic or symbolic execution slip under a correct setup.

3. E2 for a false logical inference, invalid theorem application, or invalid algebraic operation. However, if the solution deletes legal states or constructs the wrong geometry before deduction begins, the label should be E1.

4. E5 for an essential conclusion whose truth is not established because the required argument is missing.

5. E4 for an internal contradiction or answer drift within the response.

6. E3 for a repeated failed recovery loop after a contradiction is noticed.

7. `NOVEL` only when the defect genuinely fits none of the defined categories.

### 5. Quote exact evidence

For each error, copy the shortest contiguous span from the model's response that makes the error visible.

- Copy the text exactly; do not paraphrase.

- Preserve mathematical notation and delimiters.

- Do not add ellipses or truncation markers.

- Do not quote an entire paragraph when a shorter equation or sentence is sufficient.

- The span must be an actual substring of the supplied solution after JSON decoding.

- Escape the span correctly when placing it inside the output JSON string.

## Additional Reference: Error taxonomy

### C0: Verified correct reasoning

C0 describes a solution for which:

- the representation matches the problem,

- all essential deductions are valid,

- all required cases and constraints are handled,

- the requested quantity is computed, and

- the final answer is correct.

C0 is not emitted as an error code. A C0 solution has `"errors": []`.

Examples:

- A solution introduces variables for every independent side length, derives the required relations from the given constraints, checks all admissible cases, and obtains the correct AIME integer.

- A concise solution cites a standard identity whose hypotheses are visibly satisfied, performs the necessary calculation correctly, and reports the requested quantity rather than an intermediate value.

- A solution uses a different method from the reference solution but every step is valid and the final answer is equivalent to the supplied correct answer.

### E1: Fundamental representation failure

The solution encodes the problem using a model that cannot represent the given constraints or that necessarily solves a different problem.

Use E1 only for a foundational modeling or interpretation choice, usually near the beginning of the solution.

Typical cases:

- treating a general or irregular configuration as a special regular one,

- choosing variables that impose a relationship not given in the problem,

- replacing the actual sample space with a different one,

- misreading the requested object or domain in a way that determines the whole subsequent setup, or

- omitting a required degree of freedom from the initial model.

Examples:

- A geometry solution treats a non-regular equilateral hexagon as a regular hexagon from the first step, so every subsequent length relation describes the wrong figure.

- A probability solution models sampling without replacement as independent Bernoulli trials, thereby replacing the actual sample space before any counting begins.

- A number-theory solution assumes that an integer has exactly three decimal digits even though the problem permits leading zeros or representations in another base, so the chosen digit model excludes valid cases.

- A coordinate-geometry solution places three general points at $(0,0)$, $(1,0)$, and $(0,1)$ without proving that an affine normalization preserves the distances or angles used later.

Do not use E1 merely because an early equation is false. If the representation itself is adequate but a later equation or deduction is invalid, use E2 or E6.

### E2: Invalid inference or invalid operation

The solution makes a mathematically false local deduction from an otherwise usable representation.

Typical cases:

- applying a theorem without satisfying its hypotheses,

- claiming similarity, independence, symmetry, divisibility, or equality without valid grounds,

- introducing or discarding solutions through an invalid algebraic operation,

- dividing by a quantity that may be zero,

- treating an approximation as an exact equality,

- selecting the wrong requested quantity after correctly computing a different quantity, or

- asserting a conclusion that is demonstrably false from the preceding work.

Examples:

- A solution claims two triangles are similar because they share only one equal angle and no second similarity condition is established.

- From $x^2 = 25$, a solution concludes $x = 5$ even though the problem allows negative values and $x = -5$ has not been excluded.

- A solution divides an equation by $x-y$ without first ruling out the case $x=y$, thereby losing a potentially valid solution.

- A combinatorial solution multiplies two counts as though the choices were independent even though the first choice changes the number of available second choices.

- A solution computes the area correctly but reports the perimeter because it incorrectly treats that intermediate quantity as the requested target.

Use E2 rather than E5 when the asserted transition is not merely unexplained but is actually invalid.

Use E2 rather than E6 when the mistake changes the mathematical rule or method, not just its numerical execution.

### E3: Failed recovery loop

After detecting a contradiction, impossibility, or failed approach, the solution repeatedly reuses the same incompatible assumptions without resolving the problem.

Use E3 only when all of the following occur:

1. the response notices or encounters a failure,

2. it attempts to continue or repair the argument,

3. it repeats the same defective reasoning for at least one additional cycle, and

4. it provides no valid exit, restart, or corrected representation.

The `start_step` is the first step of the repeated failed recovery, not necessarily the step containing the original mathematical error.

Do not use E3 for a merely long, repetitive, or incomplete solution.

Examples:

- A geometry solution derives a negative side length, notices that the result is impossible, repeats the same false similarity assumption with renamed variables, and obtains another negative side length without changing the model.

- A probability solution obtains a value greater than $1$, acknowledges that the value cannot be a probability, recomputes the same numerator and denominator several times, and never repairs the sample space.

- An algebra solution finds that its assumed root violates the original equation, substitutes the same root again after superficial rearrangement, and cycles between the same incompatible equations.

Non-example:

- A solution abandons a failed coordinate approach and restarts with a valid synthetic argument; the original error may receive another code, but the successful restart is not E3.

### E4: Internal contradiction or answer drift

The response contains incompatible claims within its own reasoning, or its reported final answer disagrees with the value it derived immediately before.

Typical cases:

- deriving two different values for the same quantity without resolving the conflict,

- stating an equation and later using its negation,

- deriving $s = 200$ and then boxing $150$, or

- changing the final answer without a new derivation.

Examples:

- A solution derives $x=12$, later substitutes $x=10$ without explanation, and uses the substituted value to finish the calculation.

- A counting argument concludes that there are $116$ valid objects, but the final line boxes $115$ with no intervening correction.

- A solution states that two events are independent in one step and later states that the same events are dependent while continuing to use both claims.

- A solution derives $m=25$, $n=8$, and $m+n=33$, but the final line boxes $34$ without a new calculation.

Do not use E4 solely because the reported answer disagrees with the supplied reference answer. If the response consistently derives the same wrong answer, flag the earlier causal error instead.

### E5: Essential justification missing

The solution makes an essential transition whose correctness cannot be verified because the required derivation, case analysis, or justification is absent.

Typical cases:

- replacing a necessary argument with phrases such as `similarly`, `it is known`, or `by computation` when the omitted work is essential and non-routine,

- asserting an unexplained count, root set, extremum, or geometric relation,

- skipping required cases that may change the answer, or

- jumping from incomplete partial progress to a numerical result without establishing that result.

Examples:

- A solution writes "the remaining cases are similar" and reports a total count even though the omitted cases have different constraints and may contribute different numbers of objects.

- A polynomial solution lists two roots and then asserts that these are all the roots without using the degree, multiplicity, monotonicity, or another completeness argument.

- A geometry solution states that the maximum occurs in the symmetric configuration without proving symmetry, convexity, or an inequality that rules out asymmetric configurations.

- A solution derives one variable, replaces the derivations of the remaining variables with "analogously," and immediately asserts a final numerical value that cannot be checked from the written trace.

- A solution says "a computer calculation gives $237$" when the computation is essential to the answer and no algorithm, recurrence, enumeration, or verifiable intermediate result is provided.

Use E5 only when the missing material is mathematically substantive. Do not flag routine simplification, a standard one-line identity, or a concise but checkable derivation.

Use E2 instead if the asserted transition is visibly false.

### E6: Arithmetic or symbolic execution error

The overall setup and mathematical operation are correct, but their local execution contains a computational slip.

Typical cases:

- incorrect addition, multiplication, or fraction reduction,

- a sign copied incorrectly,

- an exponent, coefficient, or term dropped during an otherwise valid expansion,

- an incorrect derivative or substitution caused by a local transcription slip, or

- an incorrect simplification with the correct formula and operands.

Examples:

- A solution correctly obtains $\frac{3}{2}+\frac{2}{3}$ but simplifies it to $\frac{5}{5}$ instead of $\frac{13}{6}$.

- A valid expansion of $(a+b)^2$ is intended, but the solution accidentally drops the $2ab$ term while all preceding setup remains correct.

- After correctly deriving $-2x=5$, a solution writes $x=\frac{5}{2}$ instead of $x=-\frac{5}{2}$.

- A counting solution correctly identifies two disjoint cases of sizes $51$ and $133$ but adds them as $174$.

- A substitution uses the correct formula and values but copies an exponent as $2^5$ instead of $2^6$ in one line.

The error must be repairable locally without changing the strategy or preceding mathematical assumptions.

Use E2 instead when the operation itself is not justified.

### NOVEL: Defect outside the taxonomy

Use `NOVEL` only for a substantive defect that cannot reasonably be classified as E1, E2, E3, E4, E5, or E6.

Do not use `NOVEL` merely because you are uncertain. When using it, the quoted span must still make the defect visible.

## Final consistency checks

Before producing the response, verify all of the following:

1. `num_steps` follows the deterministic splitting rules.

2. Every `start_step` is between 1 and `num_steps`.

3. Every quoted `span` occurs exactly in the supplied solution.

4. No span is assigned more than one code.

5. No downstream consequence is duplicated after its root cause has already been flagged.

6. A wrong final answer is not assigned an additional error when an earlier root error already explains it.

7. A correct final answer does not force `errors` to be empty.

8. If `errors` is empty, the reasoning is fully valid and the final answer is correct.

9. If the final answer is wrong, the errors list identifies the earliest causal defect unless the only defect is an internally inconsistent final answer, in which case E4 is used.

## Output format

Do your analysis silently. Your visible response must be only one valid JSON object with exactly this shape:

```json
{
  "num_steps": 1,
  "errors": [
    {
      "type": "E1",
      "start_step": 1,
      "span": "<shortest exact verbatim span>"
    }
  ]
}
```

Replace the example values with the actual judgment. Do not include Markdown fences, prose, comments, confidence scores, or additional keys in the visible response. The reply must begin with `{` and end with `}`.
"""

DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-5.6-luna")
DEFAULT_PROBLEM_FILES = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "problems", "*.jsonl")))
ERROR_CODES = ["E1", "E2", "E3", "E4", "E5", "E6", "NOVEL"]


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def normalize_problem_id(pid, year=None):
    """Canonicalize AIME ids to 'YYYY-I-NN' / 'YYYY-II-NN'.

    Accepts '2024-II-4' as well as bare 1..30 indices (some solver files number a
    year's problems 1-30: 1-15 = AIME I, 16-30 = AIME II), which need `year`.
    """
    pid = str(pid).strip()
    if pid.isdigit() and year:
        n = int(pid)
        return f"{year}-I-{n:02d}" if n <= 15 else f"{year}-II-{n - 15:02d}"
    match = re.fullmatch(r"(\d{4})-(I{1,2})-(\d+)", pid)
    if not match:
        return pid
    year, part, num = match.groups()
    return f"{year}-{part}-{int(num):02d}"


def load_problems(paths=None):
    """Load problems + ground-truth answers from jsonl files.

    Each line needs at least {"id", "problem", "answer"}. Returns {problem_id: row}.
    """
    problems = {}
    for path in paths or DEFAULT_PROBLEM_FILES:
        with open(path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    problems[normalize_problem_id(row["id"])] = row
    return problems


def load_solutions(path):
    """Load solver outputs as a flat list of {"problem_idx", "sample_idx", "solution"}.

    Two input formats are supported:
      - greedy : [{"problem_idx": "2024-II-4", "solution": "..."}, ...]
                 (bare 1-30 indices are resolved using the 'aimeYYYY' in the file name)
      - samples: {"2025-I-01": ["sample 0", "sample 1", ...], ...}
    """
    with open(path) as f:
        raw = json.load(f)
    year_match = re.search(r"aime(\d{4})", os.path.basename(path))
    year = year_match.group(1) if year_match else None
    if isinstance(raw, list):
        return [
            {"problem_idx": normalize_problem_id(item["problem_idx"], year), "sample_idx": 0, "solution": item.get("solution")}
            for item in raw
        ]
    return [
        {"problem_idx": normalize_problem_id(pid), "sample_idx": i, "solution": solution}
        for pid, samples in raw.items()
        for i, solution in enumerate(samples)
    ]


def load_judgments(judgments_path, solutions_path, problems=None):
    """Join a judge output file with its solver file and the ground truth.

    Judge output files store verdicts in the same order/shape as the solver file
    (a list for greedy runs, {problem_id: [verdict per sample]} for sampled runs).
    Returns one flat record per solution:
      problem_idx, sample_idx, problem, gt_answer, solution, model_answer,
      answer_correct, num_steps, errors, judge_error
    """
    problems = problems if problems is not None else load_problems()
    solutions = load_solutions(solutions_path)
    with open(judgments_path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        verdicts = raw
    else:
        verdicts = [v for samples in raw.values() for v in samples]
    if len(verdicts) > len(solutions):
        raise ValueError(f"{judgments_path} has {len(verdicts)} verdicts but {solutions_path} only {len(solutions)} solutions")

    records = []
    for sol, verdict in zip(solutions, verdicts):
        problem = problems.get(sol["problem_idx"], {})
        model_answer = extract_model_answer(sol["solution"])
        records.append({
            "problem_idx": sol["problem_idx"],
            "sample_idx": sol["sample_idx"],
            "problem": problem.get("problem"),
            "gt_answer": problem.get("answer"),
            "solution": sol["solution"],
            "model_answer": model_answer,
            "answer_correct": answers_match(model_answer, problem.get("answer")),
            "num_steps": verdict.get("num_steps"),
            "errors": verdict.get("errors", []),
            "judge_error": verdict.get("judge_error"),
        })
    return records


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

# Some judge models quote LaTeX (e.g. \(A\), \cdot) into JSON strings without
# doubling the backslash. Walk escape tokens so a valid "\\" pair isn't
# miscounted as two lone backslashes.
_JSON_ESCAPE_TOKEN_RE = re.compile(r'\\u[0-9a-fA-F]{4}|\\["\\/bfnrtu]|\\')


def _repair_invalid_escapes(text):
    return _JSON_ESCAPE_TOKEN_RE.sub(lambda m: m.group(0) if len(m.group(0)) > 1 else "\\\\", text)


def extract_json_object(text):
    """Pull the {"num_steps": ..., "errors": [...]} object out of the judge reply,
    tolerating stray braces in surrounding text (e.g. \\boxed{220})."""
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start:i + 1]
                    try:
                        candidate = json.loads(chunk)
                    except json.JSONDecodeError:
                        try:
                            candidate = json.loads(_repair_invalid_escapes(chunk))
                        except json.JSONDecodeError:
                            break
                    if isinstance(candidate, dict) and "num_steps" in candidate and "errors" in candidate:
                        return candidate
                    break
    raise ValueError("no verdict JSON object found in judge response")


def extract_model_answer(solution_text):
    """Non-LLM extraction of the solver's final integer answer (last \\boxed{}, else last integer)."""
    if not solution_text:
        return None
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", solution_text)
    if boxed:
        match = re.search(r"-?\d+", boxed[-1])
        if match:
            return match.group(0)
    numbers = re.findall(r"-?\d+", solution_text)
    return numbers[-1] if numbers else None


def answers_match(model_answer, gt_answer):
    if model_answer is None or gt_answer is None:
        return False
    try:
        return int(model_answer) == int(gt_answer)
    except (TypeError, ValueError):
        return str(model_answer).strip() == str(gt_answer).strip()


def judge_solution(problem, gt_answer, solution, model=DEFAULT_JUDGE_MODEL, problem_idx=None, retries=3, timeout=300):
    """Judge one solution. Returns {"num_steps": int, "errors": [{"type", "start_step", "span"}, ...]}.

    Raises ValueError if the judge fails to return a parseable verdict after `retries` attempts.
    """
    import litellm

    user_content = (
        (f"problem_idx: {problem_idx}\n\n" if problem_idx else "")
        + f"Problem:\n{problem}\n\n"
        f"Correct final answer: {gt_answer}\n\n"
        f"Model's solution attempt:\n{solution}\n"
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    transient = (litellm.exceptions.Timeout, litellm.exceptions.APIConnectionError, litellm.exceptions.RateLimitError)
    kwargs = {"temperature": 0}
    last_error = None
    for attempt in range(retries):
        try:
            response = litellm.completion(model=model, messages=messages, timeout=timeout, **kwargs)
        except litellm.exceptions.BadRequestError as e:
            # some models reject the temperature param outright
            if kwargs and "temperature" in str(e).lower():
                kwargs = {}
                last_error = e
                continue
            raise
        except transient as e:
            last_error = e
            print(f"  retry {attempt + 1}/{retries} after connection error: {e}")
            continue
        raw = (response["choices"][0]["message"]["content"] or "").strip()
        try:
            return extract_json_object(raw)
        except ValueError as e:
            last_error = e
            print(f"  retry {attempt + 1}/{retries} after unparseable judge response")
    raise ValueError(f"judge call failed after {retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(args):
    from dotenv import load_dotenv

    load_dotenv()
    problems = load_problems(args.problems)
    with open(args.solutions) as f:
        is_samples = isinstance(json.load(f), dict)
    solutions = load_solutions(args.solutions)
    model_name = args.model.rsplit("/", 1)[-1]
    output_path = args.output or os.path.join("judge_outputs", f"{model_name}_judge_{os.path.basename(args.solutions)}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Resumable: verdicts already in the output file are kept and skipped.
    done = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        done = existing if isinstance(existing, list) else [v for vs in existing.values() for v in vs]
        done = done[:len(solutions)]
        # keep only the prefix before the first failure so verdicts stay aligned with solutions
        if args.retry_failed:
            done = done[:next((i for i, v in enumerate(done) if "judge_error" in v), len(done))]

    verdicts = list(done)
    for sol in solutions[len(done):]:
        problem = problems[sol["problem_idx"]]
        tag = f"{sol['problem_idx']}" + (f" sample {sol['sample_idx']}" if is_samples else "")
        if not sol["solution"]:
            verdict = {"judge_error": "empty solution"}
        else:
            try:
                verdict = judge_solution(problem["problem"], problem["answer"], sol["solution"], model=args.model, problem_idx=sol["problem_idx"])
            except ValueError as e:
                verdict = {"judge_error": str(e)}
        print(f"judged {tag}: {verdict.get('judge_error') or [e['type'] for e in verdict['errors']] or ['C0']}")
        verdicts.append(verdict)
        _write_verdicts(output_path, verdicts, solutions, is_samples)
    _write_verdicts(output_path, verdicts, solutions, is_samples)
    print(f"wrote {len(verdicts)} verdicts to {output_path}")


def _write_verdicts(path, verdicts, solutions, is_samples):
    """Write verdicts in the same shape as the solver file (list, or {problem_id: [verdicts]})."""
    if is_samples:
        out = {}
        for sol, verdict in zip(solutions, verdicts):
            out.setdefault(sol["problem_idx"], []).append(verdict)
    else:
        out = verdicts
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def export(args):
    records = load_judgments(args.judgments, args.solutions, load_problems(args.problems))
    if args.output:
        with open(args.output, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(records)} records to {args.output}")
    n_c0 = sum(1 for r in records if not r["errors"] and not r["judge_error"])
    n_correct = sum(r["answer_correct"] for r in records)
    counts = {}
    for r in records:
        for e in r["errors"]:
            counts[e["type"]] = counts.get(e["type"], 0) + 1
    print(f"{len(records)} solutions | answer correct: {n_correct} | judged C0 (no errors): {n_c0}")
    print("error counts:", dict(sorted(counts.items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="judge every solution in a solver output file")
    p_run.add_argument("--solutions", required=True, help="solver output json (greedy list or samples dict)")
    p_run.add_argument("--output", default=None, help="default: judge_outputs/{model}_judge_{solutions file}")
    p_run.add_argument("--model", default=DEFAULT_JUDGE_MODEL, help="any litellm model string")
    p_run.add_argument("--problems", nargs="+", default=None, help="problem jsonl files (default: data/problems/*.jsonl)")
    p_run.add_argument("--retry-failed", action="store_true", help="re-judge entries that previously failed")
    p_run.set_defaults(func=run)

    p_exp = sub.add_parser("export", help="join a judge output with solutions + ground truth (flat jsonl)")
    p_exp.add_argument("--solutions", required=True)
    p_exp.add_argument("--judgments", required=True)
    p_exp.add_argument("--output", default=None, help="jsonl path; omit to just print a summary")
    p_exp.add_argument("--problems", nargs="+", default=None)
    p_exp.set_defaults(func=export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
