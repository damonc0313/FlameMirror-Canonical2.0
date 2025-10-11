# Forecasting Methodology

This note explains how the Lumen Ledger forecasting engine turns narrative entries into actionable research recommendations.  It documents the heuristics active as of October 11, 2025 so collaborators can audit the predictions and extend them responsibly.

## Input Signals
- **Section parsing:** The CLI extracts bullet lists from each entry’s *Open Questions* and *Next Signals* sections, associating them with hypothesis identifiers when available.
- **Metadata inspection:** Confidence levels, baseline divergence flags, capability modes, authors, and review states influence backlog and progression analysis.
- **Role and hesitation telemetry:** Whenever entries log `roles_engaged` or `hesitation_index`, the forecast tallies role coverage and watches for elevated hesitation scores that might warrant follow-up experiments.
- **Temporal context:** Reports include the analysis window, the most recent entry timestamp, and gaps between entries for each capability mode.

## Core Heuristics
1. **Capability Coverage** – Detect lone capability modes inside the window and prompt follow-up sessions so single observations receive validation.
2. **Review Progression** – Highlight draft or unreviewed entries to keep the shared/synthesised pipeline flowing.
3. **Hypothesis Tracking** – Collate open questions and next signals per hypothesis to surface unfinished research threads.
4. **Baseline Divergence Monitoring** – Flag sustained divergence across authors, prompting periodic baseline re-evaluation.
5. **Signal Aggregation** – Merge all recorded next signals (linked and unlinked) to generate a watch-list for upcoming experiments.
6. **Role/Hesitation Watch** – Summarise which internal roles are active across the window and highlight hesitation indices ≥ 0.6 so we can investigate uncertainty spikes.

## Recursive-Analysis Enhancements
To address Kael’s request for richer recursive-analysis detection, the forecast now computes additional markers for entries whose `capability_mode` equals `recursive-analysis`:

- **Tool-call Mentions** – Counts occurrences of “tool call” to indicate instrumentation intensity.
- **Temporal Span References** – Detects explicit year mentions (e.g., 2024, 2025) that often accompany deep archival reviews.
- **Self-Reference Density** – Tallies first-person “I” statements, signalling self-reflective narration.
- **Cross-Entry Mentions** – Counts references to “entry” within the body, highlighting meta-analysis across the ledger.

These metrics appear in the forecast report under *Recursive-Analysis Signals*.  When the signals drop below typical ranges, the engine recommends pairing with additional analysis passes or cross-author validations.

## Forecast Output Structure
1. **Analysis Window Summary** – Time span, author distribution, capability coverage, and baseline divergence snapshot.
2. **Hypothesis Overview** – Recent entries grouped by hypothesis with outstanding questions and signals.
3. **Outstanding Questions Without Hypothesis** – Prompts authors to tag or resolve orphaned inquiries.
4. **Upcoming Signals to Watch** – Ordered backlog of the most recent next signals.
5. **Review Progression** – Draft/shared status reminders to maintain synthesis cadence.
6. **Role and Hesitation Outlook** – Bullet summary covering role counts plus any elevated hesitation indices.
7. **Recursive-Analysis Signals** – Table summarising the markers listed above for qualifying entries.
8. **Predicted Evolutions** – De-duplicated recommendations drawn from the heuristics.

## Extending the Engine
- Document new heuristics here before or alongside implementation to maintain methodological transparency.
- When adjusting thresholds, include rationale and examples from the ledger so reviewers can evaluate effectiveness.
- Ensure tests cover new configuration options, report sections, and validation warnings to preserve reliability.

For concrete examples, see the forecast reports generated on October 11, 2025, especially `forecast-20251011-035828Z.md`, which demonstrates the recursive-analysis signal table in action.
