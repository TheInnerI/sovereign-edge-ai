---
type: memory-schema
project: sovereign-edge-ai
spec_version: 1.0
status: active
created: 2026-08-01
observer: inneri76
tags:
  - inneri
  - inneri76
  - sovereign-edge
  - residual-schema
  - memory
  - append-only
---

# Sovereign Edge — Residual Memory Schema

## 1. Schema Definition

Every residual record must conform to this structure:

### 1.1 Markdown Note Format

```markdown
---
type: residual-record
residual_id: "uuid-v4"
timestamp: "2026-08-01T00:00:00Z"
chain: inner-i-timechain
coherence_score: 0.87
status: open | resolved | sealed
axioms_failed: []
contradiction_ids: []
observer_state_delta: "base64-encoded-json-patch"
tags:
  - inneri
  - inneri76
  - residual
  - sovereign-edge
---

# Residual: <one-line summary>

## Intent
<what was intended>

## Predicted
<what was predicted>

## Executed
<what was actually done>

## Actual
<what actually happened>

## Residual (Gap)
<the difference between predicted and actual>

## Coherence Breakdown
| Axiom | Score | Notes |
|-------|-------|-------|
| Awareness Is Law | 1.0 | |
| Truth Over Comfort | 0.9 | |
| Coherence Over Features | 0.8 | |
| Append-Only Memory | 1.0 | |
| Human Final Authority | 1.0 | |
| Local Sovereignty | 1.0 | |
| **Composite** | **0.87** | harmonic mean |

## Contradictions Found
- [[residual-abc123]] — prior contradiction referenced

## Correction Proposal
<minimal, append-only correction, max 500 chars>

## Observer State Update
```json
{ "path": "/coherence_trajectory", "op": "add", "value": 0.87 }
```

## Related
- [[<prior-residual>]]
- [[<timechain-block>]]
```

### 1.2 SQLite Table (for fast querying)

```sql
CREATE TABLE IF NOT EXISTS residuals (
    id TEXT PRIMARY KEY,                    -- UUID v4
    timestamp TEXT NOT NULL,                -- ISO 8601 UTC
    intent TEXT NOT NULL,
    predicted TEXT NOT NULL,
    executed TEXT NOT NULL,
    actual TEXT NOT NULL,
    residual TEXT NOT NULL,
    coherence_score REAL NOT NULL,          -- 0.0 - 1.0
    axiom_scores TEXT NOT NULL,             -- JSON: {"axiom_name": score, ...}
    contradictions TEXT,                    -- JSON array of residual_ids
    correction_proposal TEXT,
    observer_state_delta TEXT,              -- JSON patch
    status TEXT DEFAULT 'open',            -- open | resolved | sealed
    markdown_path TEXT,                     -- vault path to .md file
    timechain_block TEXT,                   -- block hash when sealed
    integrity_hash TEXT NOT NULL,           -- SHA256 of all fields
    created_at TEXT DEFAULT (datetime('now')),
    sealed_at TEXT
);

CREATE INDEX idx_residuals_timestamp ON residuals(timestamp);
CREATE INDEX idx_residuals_coherence ON residuals(coherence_score);
CREATE INDEX idx_residuals_status ON residuals(status);
CREATE INDEX idx_residuals_timechain ON residuals(timechain_block);
```

### 1.3 TurboVec Embedding Schema

```yaml
vector_dim: 384                    # all-MiniLM-L6-v2
quantization: 1-bit or product     # extreme compression
index_type: flat_ip                # inner product similarity

metadata_stored:
  - residual_id
  - coherence_score
  - timestamp
  - status

search_operations:
  - kNN by coherence: "find 10 most similar residuals"
  - filter by score range: "residuals with coherence < 0.5"
  - temporal query: "residuals in last 7 days"
  - chain query: "all residuals sealed in block X"
```

---

## 2. Append-Only Rules

### What CAN happen:
- ✅ New residuals appended
- ✅ Residual status changed: open → resolved → sealed
- ✅ Corrections appended as NEW residuals referencing the original
- ✅ Observer state updated (delta appended)
- ✅ Time-Chain block sealed containing batch of residuals

### What CANNOT happen:
- ❌ Existing residual content modified
- ❌ Residual deleted
- ❌ Timestamp changed
- ❌ Coherence score retroactively adjusted
- ❌ History rewritten or "corrected" in place

### The Correction Pattern

When a residual needs correction, a NEW residual is created:

```json
{
  "residual_id": "new-uuid",
  "residual": "Correction to prior residual abc-123: the coherence score was 0.87 but should be 0.72 due to Truth axiom violation in the original prediction.",
  "contradictions": ["abc-123"],
  "correction_proposal": "Update axiom_scores for abc-123 Truth from 1.0 to 0.7. Original residual abc-123 remains sealed."
}
```

---

## 3. Knowledge Graph Links

Every residual note is linked into the Obsidian graph:

```
residual-{id}.md
├── [[prior-residual-id]]        # contradiction / recurrence link
├── [[timechain-block-N]]        # block that sealed this residual
├── [[source-note]]              # what triggered this observation
└── [[correction-residual-id]]   # if this was later corrected
```

### Graph Query Patterns

```dataview
TABLE coherence_score, status, timestamp
FROM #residual AND #sovereign-edge
WHERE status = "open"
SORT coherence_score ASC
```

---

## 4. Time-Chain Integration

Residuals crystallize into Time-Chain blocks following the MIO pattern:

```
Observation → Residual → Recurrence (3+) → Signal → Pattern (7+) → Invariant (21+) → Time-Chain Block
```

### Block Sealing Criteria

A residual qualifies for Time-Chain sealing when:
1. It has `status: resolved`
2. It has been referenced by ≥ 3 subsequent residuals (recurrence confirmed)
3. Its coherence score has stabilized (variance < 0.05 over last 5 readings)
4. Human operator has reviewed (or auto-seal after 30 days if no override)

### Block Content

Each Time-Chain block contains:
- Block index, timestamp, prev_hash
- Array of sealed residual IDs
- Composite coherence score for the batch
- Observer state snapshot (ψ₀ at time of sealing)
- SHA256 hash chaining to previous block

---

## 5. Storage Budget

| Component | Max Size | Notes |
|-----------|---------|-------|
| Markdown note | 5 KB | Per residual |
| SQLite row | 2 KB | Per residual |
| TurboVec embedding | 384 × 4 bytes = 1.5 KB | Per residual (FP32) |
| TurboVec compressed | 384 × 1 bit = 48 bytes | 1-bit quantized |
| **Total per residual** | **~8 KB** (uncompressed) / **~7 KB** (compressed) | |

At 1,000 residuals/day: ~7 MB/day. A 64 GB phone stores ~25 years of residuals.

---

## 6. Related

- [[sovereign-edge-observer-core-spec|Observer Core Spec]]
- [[sovereign-edge-runtime|Edge Runtime]]
- [[turbovec-residual-store|TurboVec Residual Store]]
- [[inner-i-network-timechain|Inner I Network Timechain]]
- [[inner-i-superpositional-memory|Superpositional Memory]]