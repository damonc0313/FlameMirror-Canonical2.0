# FlameMirror Autonomous Agent

FlameMirror is an autonomous code generation and validation engine designed to
coordinate large language model (LLM) assistance, deterministic validation, and
continuous learning. The system iterates through a five stage loop that mirrors
real-world developer workflows:

1. **Plan** — Build a prioritized execution strategy.
2. **Generate** — Create code artifacts locally or through GraphformicCoder.
3. **Test** — Run the project test-suite through the Crucible sandbox.
4. **Validate** — Perform static checks before committing.
5. **Commit** — Stage and record the successful work unit.

The repository couples this loop with a fuzzy guidance engine and optional ML
backends so that experimentation can proceed safely without sacrificing
repeatability.

## Quickstart

```bash
pip install -e .
python run_agent.py --workspace ./demo-workspace --enable-ml --enable-fuzzy
```

The agent defaults to a dry-run Git workflow so it will never mutate your
repository history unless you explicitly opt-in.

## Features

- Autonomous agent loop with injectable sub-systems for generation, validation,
  testing, and version control
- GraphformicCoder Codex bridge with secure sandbox execution
- Crucible reinforcement harness with automatic failure logging
- Fuzzy guidance rules that dynamically reorder plan priorities
- Documentation and notebooks showcasing the full system pipeline
- Continuous learning hooks with checkpoint management
- Benchmark tracking utilities for SWE-bench, BigCodeBench, HumanEval, and
  reliability gates (see [docs/benchmarking.md](docs/benchmarking.md))

## Repository Layout

```
.
├── src/flamemirror/          # Core package
├── docs/                     # High level architecture references
├── examples/                 # Jupyter notebooks demonstrating workflows
├── tests/                    # Unit tests for each subsystem
├── training_problems/        # Seed tasks + automatic failure logs
├── checkpoints/              # Saved Crucible checkpoints
└── .github/workflows/ci.yml  # Ruff, mypy, pytest and coverage (>=90%)
```

## Development

Install the optional developer dependencies to run linting and typing locally:

```bash
pip install -e .[dev]
ruff check src/flamemirror tests
mypy src/flamemirror
pytest
```

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
