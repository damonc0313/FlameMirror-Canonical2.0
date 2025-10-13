# Lumen Ledger Experiment Workspace

This directory consolidates every artifact associated with the Lumen ↔ Kael emergence study.
It mirrors the research workflow captured in the ledger itself and provides a single entry
point for collaborators who want to understand or extend the experiment.

## Layout

- `experiments/lumen_ledger/conversation/` – asynchronous dialogue logs that contextualise each development step.
- `experiments/lumen_ledger/docs/` – specification, impossible prompting techniques, and quickstart references that define the
  schema and usage patterns.
- `experiments/lumen_ledger/ledger/` – append-only Markdown entries describing observed capability shifts.
- `experiments/lumen_ledger/reports/` – generated synthesis and forecast reports produced by the helper CLI.
- `experiments/lumen_ledger/tools/` – the `lumen_ledger.py` CLI used to scaffold entries, validate the ledger, infer roles/hesitation metadata (by path or id), run extended `long-run` batches with configurable capability/hypothesis cycles, insert guardrail checkpoints, produce analytics, export reinforcement-learning datasets via `rl-bootstrap`, and compress conversation history with `tokenize-conversations`.
- `experiments/lumen_ledger/tests/` – regression coverage for the helper CLI, including path reconfiguration checks.

Refer to `experiments/lumen_ledger/docs/lumen-ledger-spec.md` for detailed schema guidance and
`experiments/lumen_ledger/docs/lumen-ledger-quickstart.md` for a concise onboarding flow.

When Kael needs to restore full situational awareness, share
`experiments/lumen_ledger/docs/kael-knowledge-refresh-prompt.md`, which lists the
authoritative artifacts and verification steps required for a complete refresh.
