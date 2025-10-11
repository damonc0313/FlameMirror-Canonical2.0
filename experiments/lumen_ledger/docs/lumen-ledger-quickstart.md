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

## Optional: Infer Roles and Hesitation Automatically
When you want the ledger to capture which internal roles were active (planner, experimenter, guardrail) or to log linguistic hesitation, run the analyzer against the entry you just wrote:

```bash
python experiments/lumen_ledger/tools/lumen_ledger.py analyze --path experiments/lumen_ledger/ledger/2025/10/11/20251011-your-entry.md --apply
```

The command prints the inferred roles and hesitation index to stdout and, with `--apply`, writes them into the frontmatter so future syntheses and forecasts can track role coverage and uncertainty trends.

## 5. Review and Iterate
After writing an entry:

1. Share the file with your counterpart for review.
2. Update the frontmatter fields `review_state`, `reviewed_by`, and `review_notes` when feedback is addressed.
3. Re-run `validate` before committing so the index stays current.

For a deeper reference, see [`experiments/lumen_ledger/docs/lumen-ledger-spec.md`](./lumen-ledger-spec.md), which details the schema, evolution guidelines, and research cadence.
