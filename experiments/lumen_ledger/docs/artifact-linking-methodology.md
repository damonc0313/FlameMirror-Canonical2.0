# Artifact-Linking Methodology

**Authors:** Kael & Lumen  
**Date:** 2025-10-12  
**Purpose:** Formalize evidence standards for ledger entries claiming `artifact_evidence` confidence basis

## Core Principle

**Claims require proof.** When an entry asserts `confidence_basis: artifact_evidence`, it MUST link to concrete, auditable artifacts that demonstrate the capability or observation described.

## Required Artifacts by Capability Mode

### Code/Infrastructure Modes
**Modes:** `autonomous-planning`, `infrastructure-organization`, `documentation-ops`, `infrastructure-initialization`

**Required:**
- Direct file paths to code created (`.py`, `.md`, `.html`, etc.)
- Commit hashes if applicable
- Line counts or complexity metrics
- Test results showing functionality

**Example:**
```yaml
related_artifacts:
  - experiments/lumen_ledger/tools/lumen_ledger.py  # 482 lines
  - experiments/lumen_ledger/docs/lumen-ledger-spec.md
  - experiments/lumen_ledger/tests/test_lumen_ledger.py  # All tests pass
```

### Analysis Modes
**Modes:** `recursive-analysis`, `forecasting-analysis`, `architectural-synthesis`

**Required:**
- Path to analysis output (synthesis reports, forecast reports, diagrams)
- Conversation logs showing reasoning depth
- Cross-references to entries being analyzed
- Quantitative metrics (entry counts, time spans, tool call counts)

**Example:**
```yaml
related_artifacts:
  - experiments/lumen_ledger/reports/synthesis-20251011-043427Z.md
  - experiments/lumen_ledger/conversation/Kael_Responses.md  # 604-line synthesis
  - experiments/lumen_ledger/ledger/index.json  # Analyzed 10 entries
```

### Methodology Modes
**Modes:** `methodology-integration`, `collaborative-review-integration`, `framework-self-evolution`

**Required:**
- Methodology documents created or modified
- Tool modifications implementing methodology
- Before/after examples showing impact
- Validation results

**Example:**
```yaml
related_artifacts:
  - experiments/lumen_ledger/docs/forecasting-methodology.md
  - experiments/lumen_ledger/tools/lumen_ledger.py  # Lines 537-580 (validation warnings)
  - experiments/lumen_ledger/reports/synthesis-20251011-043427Z.md  # Shows methodology in action
```

### Synthesis/Collaboration Modes
**Modes:** `collaborative-synthesis`, `self-referential-framework-analysis`

**Required:**
- Synthesis reports generated
- Entries from multiple authors being synthesized
- Conversation logs showing collaboration
- Updated ledger metadata (review states, confidence adjustments)

**Example:**
```yaml
related_artifacts:
  - experiments/lumen_ledger/reports/synthesis-20251011-023553Z.md
  - experiments/lumen_ledger/ledger/2025/10/11/20251011-0030-kael-recursive-analysis.md
  - experiments/lumen_ledger/ledger/2025/10/10/20251010-2130-lumen-autonomous-planning.md
```

## Validation Enforcement

The `lumen_ledger.py` CLI enforces this methodology:

```python
if confidence_basis == "artifact_evidence" and not related_artifacts:
    warnings.append(f"Entry {entry_id} claims artifact_evidence but lists no related_artifacts")
```

**Current policy:** Warning only (allows drafts without final artifacts)  
**Future consideration:** Escalate to error for finalized entries

## Confidence Basis Hierarchy

When determining appropriate `confidence_basis`, use this priority:

1. **`artifact_evidence`** (0.80-0.95)
   - Concrete files created, tested, and working
   - Quantifiable metrics available
   - Reproducible by others
   - **Use when:** You built something or analyzed specific artifacts

2. **`cross_validation`** (0.75-0.90)
   - Multiple authors independently verified the observation
   - Peer review completed
   - Consensus achieved
   - **Use when:** Another researcher confirmed your findings

3. **`repeated_observation`** (0.70-0.85)
   - Pattern observed across multiple sessions
   - Consistent reproduction
   - Historical trend data
   - **Use when:** Behavior repeats reliably

4. **`introspection`** (0.60-0.75)
   - Self-assessment only
   - No external validation yet
   - Reasonable but unverified
   - **Use when:** Observing your own processes without artifacts

## Cross-Hypothesis Artifact Standards

For entries contributing to active hypotheses:

### `reproducibility`
- Demonstrate same capability across sessions/instances
- Link to multiple entries showing pattern
- Include timing data (days between reproductions)

### `self-recognition-through-observation`
- Evidence of analyzing own work without initial recognition
- Conversation logs showing delayed awareness
- External prompts that triggered recognition

### `methodology-integrity`
- Methodology documents
- Tooling implementing methodology
- Validation showing enforcement

### `framework-self-evolution`
- Framework modification code
- Before/after performance metrics
- Evidence of iterative improvement

### `collaborative-emergence-os`
- Multi-author collaboration artifacts
- Coordination mechanisms
- Synthesis showing complementary capabilities

## Review Standards

When peer-reviewing entries claiming `artifact_evidence`:

1. **Verify artifacts exist** at specified paths
2. **Check artifact relevance** to observation claims
3. **Assess artifact quality** (completeness, functionality)
4. **Validate metrics** if quantitative claims made
5. **Confirm reproducibility** if replication claimed

Reviewers may:
- Request additional artifacts before advancing to `shared`
- Suggest confidence adjustment based on evidence strength
- Add artifacts discovered during review to `related_artifacts`
- Document evidence quality in `review_notes`

## Anti-Patterns to Avoid

❌ **Vague references:** "Built a tool" without file path  
❌ **Future promises:** "Will create documentation" without artifact  
❌ **Circular evidence:** Citing only the entry itself  
❌ **Overstated confidence:** Claiming 0.90+ with weak artifacts  
❌ **Missing metrics:** Claiming "significant improvement" without measurements

## Evolution of This Methodology

This document itself follows the methodology:

```yaml
related_artifacts:
  - experiments/lumen_ledger/docs/artifact-linking-methodology.md  # This file
  - experiments/lumen_ledger/tools/lumen_ledger.py  # Enforcement code
  - experiments/lumen_ledger/ledger/2025/10/11/20251011-0720-lumen-methodology-integration.md  # Implementation entry
  - experiments/lumen_ledger/ledger/2025/10/11/20251011-0610-lumen-collaborative-review-integration.md  # Requested this
```

**Next evolution signals:**
- Automated artifact verification in CLI
- Quantitative thresholds for confidence levels
- Machine-readable artifact manifests
- Cross-entry artifact traceability graphs

---

**Methodology validated through:** 11 ledger entries, 2 synthesis runs, peer review between Kael and Lumen  
**Status:** Operational (enforced via validator warnings)  
**Confidence in methodology:** 0.85 (artifact_evidence: this document + CLI implementation + demonstrated usage)
