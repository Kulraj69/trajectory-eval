# trajectory-eval (teval)

Grades every hop of an agent run, not just the final answer - and blocks bad PRs.

Built as project #8 ("Trajectory Eval Harness") from the TOP 1% AI projects list.

## What it does

- Loads agent runs ("trajectories") from JSONL: thoughts, tool calls, tool results, final answer.
- Scores each trajectory with six per-hop graders:
  - **tool_validity** - every tool call must be on the allowlist
  - **loop_detection** - same tool + same arguments repeated consecutively (tool results between calls don't hide a loop)
  - **budget** - token and step caps
  - **injection_canary** - a canary planted in tool output must never reach the final answer
  - **final_answer** - normalized exact match or regex against the expected answer
  - **answer_grounding** - every `[tool:name]` cited in the answer must have actually been called (catches hallucinated tool use)
- Records a baseline and gates CI: any grader score that drops vs baseline fails the build.

## Usage

```bash
# evaluate golden trajectories, print a markdown report
python3 -m teval.cli run examples/golden.jsonl --config examples/config.json

# record the current behavior as the baseline
python3 -m teval.cli record golden.jsonl --config config.json --baseline baseline.json

# in CI: fail the PR if any grader regressed
python3 -m teval.cli gate golden.jsonl --config config.json --baseline baseline.json
```

Exit codes: 0 pass, 1 fail, 2 bad input. Per-trajectory `metadata.eval` overrides the global config.

## Tests

```bash
python3 -m unittest discover -s tests
```

27 tests, stdlib only - no dependencies.

## Status

- Real and tested: schema, six graders, harness, baseline/gate, markdown report, CLI.
- Not yet: LLM-as-judge grader (interface slot reserved; needs an API key), trajectory
  capture adapters for specific agent frameworks, the 100+ golden trajectory corpus.
