---
type: infrastructure-spec
project: sovereign-edge-ai
spec_version: 1.0
status: active
created: 2026-08-01
observer: inneri76
tags:
  - inneri
  - inneri76
  - sovereign-edge
  - turbovec
  - vector-store
  - embeddings
---

# TurboVec Residual Store — Extreme Vector Compression

## 1. Purpose

Store residual embeddings at extreme compression ratios for phone-class hardware. Every residual observation gets:
1. A full-text Markdown note in the vault
2. A TF-IDF sparse vector for exact keyword matching
3. A 1-bit quantized dense embedding for semantic search

---

## 2. Architecture

```
Residual Text
      │
      ├──→ Markdown Note (vault)         ← permanent source of truth
      │
      ├──→ TF-IDF Sparse Vector          ← keyword search (scikit-learn)
      │     └── sqlite FTS5 full-text index
      │
      └──→ Dense Embedding (384d)        ← semantic search
            │
            └──→ 1-bit Quantize           ← extreme compression
                  └──→ Binary Index (flat IP)
```

### Why Not FAISS/Chroma/Pinecone?

- **FAISS**: C++ dependency, complex build on Android
- **Chroma**: Requires Python, heavy for phones
- **Pinecone**: Cloud dependency (violates sovereignty)
- **TurboVec approach**: Pure Python + numpy, 1-bit quantization, flat index. Runs anywhere.

---

## 3. Implementation (Pure Python, Zero Deps Beyond numpy)

```python
"""
TurboVec — Extreme Vector Compression for Edge Devices
No FAISS, no Chroma, no cloud. Pure numpy + bit packing.
"""
import numpy as np
import json
import sqlite3
from pathlib import Path


class TurboVecStore:
    """1-bit quantized vector store for residual embeddings."""

    def __init__(self, db_path: str = "turbovec.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                residual_id TEXT PRIMARY KEY,
                embedding_bits BLOB NOT NULL,       -- packed 1-bit vector
                tfidf_tokens TEXT,                   -- JSON: {token: weight}
                coherence_score REAL,
                timestamp TEXT,
                FOREIGN KEY (residual_id) REFERENCES residuals(id)
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_vec_coherence ON vectors(coherence_score)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_vec_timestamp ON vectors(timestamp)")

    def quantize_1bit(self, vec: np.ndarray) -> bytes:
        """
        Quantize float32 embedding to 1-bit.
        Positive values → 1, negative/zero → 0.
        Packs 8 bits per byte.
        """
        bits = (vec > 0).astype(np.uint8)
        packed = np.packbits(bits)
        return packed.tobytes()

    def dequantize_1bit(self, packed: bytes, dim: int = 384) -> np.ndarray:
        """Unpack 1-bit vector back to approximate float32 (1.0 or -1.0)."""
        bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
        bits = bits[:dim]  # trim padding bits
        return bits.astype(np.float32) * 2.0 - 1.0  # map 0→-1, 1→1

    def add(self, residual_id: str, embedding: np.ndarray,
            coherence_score: float, timestamp: str, tfidf_tokens: dict = None):
        """Store a quantized embedding."""
        packed = self.quantize_1bit(embedding)
        tokens_json = json.dumps(tfidf_tokens) if tfidf_tokens else "{}"
        self.conn.execute(
            "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?, ?)",
            (residual_id, packed, tokens_json, coherence_score, timestamp)
        )
        self.conn.commit()

    def search(self, query_embedding: np.ndarray, k: int = 10) -> list:
        """
        Brute-force inner product search over 1-bit embeddings.
        Fast enough for <100K vectors on phone hardware (~5ms for 10K vectors).
        """
        query_q = self.quantize_1bit(query_embedding)
        query_bits = np.unpackbits(np.frombuffer(query_q, dtype=np.uint8))[:384]

        cursor = self.conn.execute("SELECT residual_id, embedding_bits, coherence_score FROM vectors")
        results = []
        for row in cursor:
            stored_bits = np.unpackbits(np.frombuffer(row[1], dtype=np.uint8))[:384]
            # Hamming similarity via inner product on {-1, 1}
            stored_vec = stored_bits.astype(np.float32) * 2.0 - 1.0
            query_vec = query_bits.astype(np.float32) * 2.0 - 1.0
            similarity = float(np.dot(query_vec, stored_vec) / 384.0)  # normalize
            results.append((row[0], similarity, row[2]))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def get_stats(self) -> dict:
        """Storage efficiency stats."""
        count = self.conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        db_size = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
        return {
            "total_vectors": count,
            "db_size_bytes": db_size,
            "bytes_per_vector": db_size / count if count > 0 else 0,
            "compression_ratio": (384 * 4) / (384 / 8),  # float32 vs 1-bit: ~32x
        }
```

---

## 4. TF-IDF + SVD Fallback (Zero GPU Required)

For devices where dense embeddings are too slow, use TF-IDF + truncated SVD:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class LightweightEmbedder:
    """TF-IDF → SVD → 1-bit quantize. No PyTorch, no GPU."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        self.svd = TruncatedSVD(n_components=dim, random_state=42)
        self.fitted = False

    def fit(self, texts: list[str]):
        X = self.tfidf.fit_transform(texts)
        self.svd.fit(X)
        self.fitted = True

    def embed(self, text: str) -> np.ndarray:
        X = self.tfidf.transform([text])
        vec = self.svd.transform(X)[0]
        # Normalize to unit length
        return vec / (np.linalg.norm(vec) + 1e-8)
```

This produces 384-dim embeddings with **zero GPU, zero PyTorch** — pure scikit-learn. Train once, embed forever.

---

## 5. Phone Storage Budget

| Component | Per Vector | 10K Vectors | 100K Vectors |
|-----------|-----------|-------------|--------------|
| 1-bit embedding | 48 bytes | 480 KB | 4.8 MB |
| TF-IDF tokens (avg) | 100 bytes | 1 MB | 10 MB |
| SQLite overhead | ~50 bytes | 500 KB | 5 MB |
| **Total** | **~200 bytes** | **~2 MB** | **~20 MB** |

Compare: FAISS IndexFlatIP (float32, 384d) = 1.5 KB per vector → 150 MB for 100K vectors.
TurboVec: **~75x smaller**.

---

## 6. Search Performance (Estimated)

| Vectors | Brute-Force Time | Memory |
|---------|-----------------|--------|
| 1,000 | < 1 ms | 200 KB |
| 10,000 | ~5 ms | 2 MB |
| 100,000 | ~50 ms | 20 MB |
| 1,000,000 | ~500 ms | 200 MB |

For >1M vectors, switch to IVF (inverted file) index with 1-bit quantization — stays at ~200 MB.

---

## 7. Integration with Observer Core

```python
from turbovec_store import TurboVecStore
from sentence_transformers import SentenceTransformer  # or LightweightEmbedder

# Initialize
store = TurboVecStore("residuals_turbovec.db")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 384d, 80 MB

# On each residual detection
def on_residual(residual_json: dict):
    # 1. Embed the residual text
    text = f"{residual_json['intent']} {residual_json['actual']} {residual_json['residual']}"
    embedding = embedder.encode(text)

    # 2. Store 1-bit quantized
    store.add(
        residual_id=residual_json["residual_id"],
        embedding=embedding,
        coherence_score=residual_json["coherence_score"],
        timestamp=residual_json["timestamp"]
    )

    # 3. Write Markdown note (separate vault write)

    # 4. Check for similar past residuals
    similar = store.search(embedding, k=5)
    for rid, sim, score in similar:
        if sim > 0.95:
            # Possible recurrence — flag for invariant tracking
            pass
```

---

## 8. Related

- [[sovereign-edge-observer-core-spec|Observer Core Spec]]
- [[sovereign-edge-runtime|Edge Runtime]]
- [[sovereign-edge-residual-schema|Residual Schema]]
- [[inner-i-network-timechain|Time-Chain Architecture]]