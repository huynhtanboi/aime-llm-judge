# AIME LLM Judge

Step-level error annotation of LLM solutions to AIME problems.

Given a **problem**, a **solver's solution** and the **ground-truth answer**, an LLM judge
splits the solution into steps and labels every independent reasoning error with:

| field        | meaning                                                              |
|--------------|----------------------------------------------------------------------|
| `type`       | error type: `E1`–`E6` or `NOVEL` (no errors at all = `C0`)            |
| `start_step` | 1-based step where the error first becomes visible                   |
| `span`       | shortest verbatim quote from the solution that shows the error       |

Example verdict:

```json
{"num_steps": 8,
 "errors": [{"type": "E6", "start_step": 2, "span": "\\frac{1}{x^2 y^2 z^2} = 2^{1/2 + 1/3 + 1/4}"},
            {"type": "E2", "start_step": 5, "span": "x^4 y^3 z^2 = (xyz)^4 \\cdot (xyz)^3 \\cdot (xyz)^2"}]}
```

## Error taxonomy

| Code    | Name                                              |
|---------|---------------------------------------------------|
| C0      | verified correct reasoning (`"errors": []`)       |
| E1      | fundamental representation failure                |
| E2      | invalid inference or invalid operation            |
| E3      | failed recovery loop                              |
| E4      | internal contradiction or answer drift            |
| E5      | essential justification missing                   |
| E6      | arithmetic or symbolic execution error            |
| NOVEL   | defect outside the taxonomy                       |

Full definitions, examples and the decision order are in `JUDGE_SYSTEM_PROMPT` in [judge.py](judge.py).

## Repository layout

```
judge.py                         LLM judge + data loader + CLI
data/
  problems/                      problems + ground-truth answers (jsonl: id, problem, answer, ...)
    aime_2024_2026.jsonl
    aime_2018_2020.jsonl
    hf_original/aime_2024_2026_hf.jsonl   HF/MathArena wording (used for the greedy runs, see below)
  solutions/
    greedy/                      one solution per problem:  [{"problem_idx", "solution"}, ...]
    samples/                     several samples per problem: {"2025-I-01": ["...", ...], ...}
judge_outputs/
  greedy/                        gpt-5.6-luna verdicts for data/solutions/greedy/*
  samples/                       gpt-5.6-luna verdicts for data/solutions/samples/*
  multi_judge_legacy_rubric/     5 judge models on qwen2.5 (older E1–E7 rubric, see docs/)
docs/legacy_rubric_E1-E7.md      the older rubric used by multi_judge_legacy_rubric/
```

Judge output files are stored **in the same order/shape as their solver file** (a list for greedy
runs, `{problem_id: [verdict per sample]}` for sampled runs) and contain only what the judge
returned (`num_steps`, `errors`) or `{"judge_error": ...}` if the call failed. Use the data
loader to join them with problems, ground truth and solutions.

Solver files: `qwen2.5_*` = Qwen2.5-Instruct (1.5B / 7B / 32B; no size = earlier 7B run),
`qwen3.5_*` = Qwen3.5 (2B / 9B / 35B). File `{judge}_judge_{solver file}` judges `{solver file}`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in the API key(s) for your judge model
```

The judge is called through [litellm](https://github.com/BerriAI/litellm), so any litellm model
string works (`openai/...`, `anthropic/...`, `ollama/...`). Default: `openai/gpt-5.6-luna`
(override with `--model` or the `JUDGE_MODEL` env var).

## Usage

### Run the judge

```bash
# greedy solutions -> judge_outputs/gpt-5.6-luna_judge_qwen2.5_7b_aime2024.json
python judge.py run --solutions data/solutions/greedy/qwen2.5_7b_aime2024.json

# sampled solutions, different judge, custom output path
python judge.py run --solutions data/solutions/samples/qwen3.5_9b_samples.json \
    --model anthropic/claude-opus-4-8 --output judge_outputs/my_run.json
```

Runs are resumable: re-running the same command skips verdicts already in the output file
(`--retry-failed` re-judges from the first failed entry).

To judge your own solver, write its outputs in either solution format above; problem ids may be
`2025-I-01`, `2025-I-1`, or a bare `1`–`30` index if the file name contains `aimeYYYY`.

### Load / inspect judge outputs

```bash
python judge.py export \
    --solutions data/solutions/greedy/qwen2.5_7b_aime2024.json \
    --judgments judge_outputs/greedy/gpt-5.6-luna_judge_qwen2.5_7b_aime2024.json \
    --output qwen2.5_7b_aime2024_annotated.jsonl
# 30 solutions | answer correct: 4 | judged C0 (no errors): 3
# error counts: {'E1': 12, 'E2': 37, 'E4': 3, 'E5': 3, 'E6': 10}
```

From Python:

```python
from judge import load_problems, load_solutions, load_judgments, judge_solution

records = load_judgments(
    "judge_outputs/greedy/gpt-5.6-luna_judge_qwen2.5_7b_aime2024.json",
    "data/solutions/greedy/qwen2.5_7b_aime2024.json",
)
r = records[0]
r["problem_idx"], r["gt_answer"], r["model_answer"], r["answer_correct"]
r["errors"]   # [{"type": "E6", "start_step": 2, "span": "..."}, ...]

# judge a single solution directly
verdict = judge_solution(problem="...", gt_answer="033", solution="...", model="openai/gpt-5.6-luna")
```

Each record has: `problem_idx, sample_idx, problem, gt_answer, solution, model_answer,
answer_correct, num_steps, errors, judge_error`. `answer_correct` is computed deterministically
(last `\boxed{}` / last integer vs ground truth), not by the judge.

## Notes

- **Problem wording.** The greedy 2024–2026 runs were judged with the HF / MathArena problem text
  (`data/problems/hf_original/`); the sampled runs with `data/problems/aime_2024_2026.jsonl`.
  The two sources word several problems differently and disagree on one answer
  (2026-II-10: 850 vs 340). Pass `--problems data/problems/hf_original/aime_2024_2026_hf.jsonl`
  to reproduce greedy-run numbers exactly.
- Spans are produced by an LLM; a small fraction are not exact substrings of the solution.
- `multi_judge_legacy_rubric/` uses the older E1–E7 codes (E5/E6/E7 mean different things there);
  don't mix its counts with the E1–E6 outputs. `claude-sonnet-4-6` 2024 and `gpt-5.2` 2025 there
  are partial runs.
- Data licenses: AIME 2025/2026 problems from MathArena are CC BY-NC-SA 4.0.
