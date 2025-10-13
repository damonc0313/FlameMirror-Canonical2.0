# Lumen Ledger Quickstart

This primer walks through the minimum steps required to start using the Lumen Ledger tooling. It is intentionally concise so new collaborators can begin logging observations without reading the full specification immediately.

## 1. Validate the Workspace
Run the validator to ensure any existing entries conform to the schema and the metadata index is up to date:

```bash
python experiments/lumen_ledger/tools/lumen_ledger.py validate
```

If issues are reported, address them before adding new data. A clean validation run regenerates `experiments/lumen_ledger/ledger/index.json`.

## 2. Create a New Entry
Scaffold an entry using the helper CLI. Supply frontmatter fields on the command line to minimise interactive prompts:

```bash
python experiments/lumen_ledger/tools/lumen_ledger.py init-entry \
  --author lumen \
  --timestamp 2025-10-11T05:15:00Z \
  --catalyst "Co-run planning" \
  --capability-mode "collaborative-planning" \
  --hypothesis-id reproducibility \
  --confidence 0.75 \
  --confidence-basis introspection \
  --baseline-divergence
```

Omit `--no-editor` to open the scaffolded file in the editor defined by `$VISUAL` or `$EDITOR`. Use `--dry-run` first if you want to preview the generated path and frontmatter without writing to disk.

### Optional Metadata
- `--related-artifacts`: reference commits, transcripts, or supporting files.
- `--review-state`, `--reviewed-by`, `--review-notes`: capture peer-review progress.
- `--force`: overwrite an existing file if you are intentionally revising it.

## 3. Populate the Entry
Each generated file contains the required Markdown sections:

- `## Observation`
- `## Evidence`
- `## Delta`
- `## Open Questions`
- `## Next Signals`

Replace the placeholder bullets with concise notes. When linking evidence, include repository-relative paths or commit hashes to keep the ledger auditable.

## 4. Run Synthesis or Forecast Reports
Generate aggregate views to understand current trends. Both commands accept filtering flags (`--since`, `--author`, `--mode`, `--hypothesis`) and a `--limit` cap:

```bash
python experiments/lumen_ledger/tools/lumen_ledger.py synthesize --limit 5
python experiments/lumen_ledger/tools/lumen_ledger.py forecast --since 2025-10-10T21:00:00Z --author kael
```

Reports are written to `experiments/lumen_ledger/reports/` and the console prints headline statistics so you can confirm the slice that was analysed.

## Optional: Launch a Long Autonomous Run
When you want the helper to produce a full batch of entries in a single action—for example, to stress-test autonomous cadence—use the `long-run` command. It rotates through common capability modes, writes each entry with pre-populated sections, refreshes the index, and generates fresh synthesis/forecast reports for the same window.

```bash
python experiments/lumen_ledger/tools/lumen_ledger.py long-run \
  --duration-hours 6 \
  --interval-minutes 30 \
  --author lumen \
  --resume
```

You can supply `--steps` explicitly or let the helper derive it from `--duration-hours`. Use `--resume` to pick up the capability/hypothesis rotation where the previous batch ended, which keeps long autonomous runs coherent without manually tracking offsets. `--confidence-step` and `--hesitation-step` expose the heuristics that drive incremental metadata during extended sessions, while `--no-reports` skips synthesis/forecast generation when you just need the entries.

Longer unattended runs benefit from the new cadence controls:

- `--capability-cycle` and `--hypothesis-cycle` accept comma-separated lists so you can extend or customise the rotation when the default six-mode loop is not sufficient.
- `--governance-cadence` inserts an additional guardrail checkpoint entry after every _N_ segments. These checkpoints automatically cite the standing permission note and keep the guardrail voice visible in the ledger index, which helps sustain autonomy without tripping safety thresholds.

For example, to run a twelve-hour session that inserts a governance checkpoint every three segments while using a bespoke capability rotation:

```bash
python experiments/lumen_ledger/tools/lumen_ledger.py long-run \
  --duration-hours 12 \
  --interval-minutes 45 \
  --capability-cycle "autonomous-planning,recursive-analysis,architecture-scout" \
  --governance-cadence 3
```

## Optional: Export an RL Dataset
When you want to feed past ledger activity into a reinforcement-learning experiment, use `rl-bootstrap` to generate a structured dataset. The command respects the same filtering flags as `synthesize` and `forecast`, and writes a JSON payload with reward estimates, feature counts, and aggregate statistics.

```bash
python experiments/lumen_ledger/tools/lumen_ledger.py rl-bootstrap \
  --since 2025-10-10T21:00:00Z \
  --limit 50 \
  --weights confidence=0.6 reviewer_bonus=0.1 \
  --print-summary
```

By default the dataset is written to `experiments/lumen_ledger/reports/rl-dataset-<timestamp>.json`. Supply `--output` to point the export at another directory, for example when you want a model-training job to pick it up automatically.

## Optional: Compress Conversation History
The `tokenize-conversations` command builds a reusable token dictionary from the asynchronous dialogue logs. It compresses each conversation file, writes a JSON index of the compressed content, and generates a markdown report summarising token coverage and compression ratios.

```bash
python experiments/lumen_ledger/tools/lumen_ledger.py tokenize-conversations \
  --input experiments/lumen_ledger/conversation \
  --output-dir experiments/lumen_ledger/conversation/tokenizer \
  --min-frequency 5 \
  --max-tokens 250
```

Omit the optional flags to use the defaults. The command creates `conversation_tokens.json`, `compressed_conversations.json`, and a `tokenizer-report.md` file inside the output directory so downstream tooling can reuse the vocabulary without reprocessing the corpus.

## Optional: Infer Roles and Hesitation Automatically
When you want the ledger to capture which internal roles were active (planner, experimenter, guardrail) or to log linguistic hesitation, run the analyzer against the entry you just wrote:

```bash
python experiments/lumen_ledger/tools/lumen_ledger.py analyze \
  --path experiments/lumen_ledger/ledger/2025/10/11/20251011-your-entry.md \
  --apply --refresh-index
```

The command prints the inferred roles and hesitation index to stdout and, with `--apply`, writes them into the frontmatter so future syntheses and forecasts can track role coverage and uncertainty trends. Pass `--refresh-index` to rebuild the shared `index.json` after applying metadata so other commands immediately pick up the new values.

If you already validated the workspace and just need to revisit an entry later, you can resolve it by identifier instead of typing the path:

```bash
python experiments/lumen_ledger/tools/lumen_ledger.py analyze --id 20251011-your-entry --apply
```

The helper looks up the identifier in the index and updates the corresponding file.

## 5. Review and Iterate
After writing an entry:

1. Share the file with your counterpart for review.
2. Update the frontmatter fields `review_state`, `reviewed_by`, and `review_notes` when feedback is addressed.
3. Re-run `validate` before committing so the index stays current.

For a deeper reference, see [`experiments/lumen_ledger/docs/lumen-ledger-spec.md`](./lumen-ledger-spec.md), which details the schema, evolution guidelines, and research cadence.
