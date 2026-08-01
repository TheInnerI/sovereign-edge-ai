"""
Lightweight Embedder — TF-IDF + TruncatedSVD

Zero GPU. Zero PyTorch. Pure scikit-learn.
Produces 384-dim embeddings from text.
Fits on CPU in seconds. Embeds in microseconds.
"""
import numpy as np
import pickle
from pathlib import Path
from typing import Optional


class LightweightEmbedder:
    """
    TF-IDF → TruncatedSVD → L2 normalize.
    No GPU, no PyTorch, no sentence-transformers.

    Train once on a corpus of residual texts, then embed forever.
    """

    def __init__(self, dim: int = 384, cache_dir: str = "data/"):
        self.dim = dim
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tfidf = None
        self.svd = None
        self.fitted = False

    def fit(self, texts: list[str], save: bool = True):
        """Fit TF-IDF + SVD on a corpus of texts."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        print(f"Fitting TF-IDF on {len(texts)} texts...")
        self.tfidf = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        X = self.tfidf.fit_transform(texts)
        print(f"Vocabulary size: {len(self.tfidf.vocabulary_)}")

        n_components = min(self.dim, X.shape[1] - 1, X.shape[0] - 1)
        print(f"Fitting TruncatedSVD to {n_components} dims...")
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(X)
        self.fitted = True
        print(f"Explained variance: {self.svd.explained_variance_ratio_.sum():.2%}")

        if save:
            self.save()
        return self

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text → 384-dim normalized vector."""
        if not self.fitted:
            raise RuntimeError("Embedder not fitted. Call .fit(corpus) first.")
        X = self.tfidf.transform([text])
        vec = self.svd.transform(X)[0]
        # Pad or truncate to self.dim
        if len(vec) < self.dim:
            vec = np.pad(vec, (0, self.dim - len(vec)), 'constant')
        else:
            vec = vec[:self.dim]
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts."""
        X = self.tfidf.transform(texts)
        vecs = self.svd.transform(X)
        if vecs.shape[1] < self.dim:
            vecs = np.pad(vecs, ((0, 0), (0, self.dim - vecs.shape[1])), 'constant')
        else:
            vecs = vecs[:, :self.dim]
        # L2 normalize rows
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (vecs / norms).astype(np.float32)

    def save(self):
        """Save fitted embedder to disk."""
        path = self.cache_dir / "lightweight_embedder.pkl"
        with open(path, 'wb') as f:
            pickle.dump({
                'tfidf': self.tfidf,
                'svd': self.svd,
                'dim': self.dim,
                'fitted': self.fitted,
            }, f)
        print(f"Saved embedder to {path}")

    @classmethod
    def load(cls, cache_dir: str = "data/"):
        """Load a pre-fitted embedder."""
        path = Path(cache_dir) / "lightweight_embedder.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No cached embedder at {path}. Fit one first.")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        inst = cls(dim=data['dim'], cache_dir=cache_dir)
        inst.tfidf = data['tfidf']
        inst.svd = data['svd']
        inst.fitted = data['fitted']
        return inst


# ─── Pretraining corpus (residual detection language) ─────────────────

PRETRAIN_CORPUS = [
    # Intent/action/outcome patterns
    "intent was to generate truthful product description but output contained unsubstantiated claims",
    "predicted coherence score 0.85 but actual output had fabrication in third paragraph",
    "system intended to summarize document accurately but hallucinated statistics not present in source",
    "expected user satisfaction improved but actual feedback showed confusion about pricing",
    "model predicted high accuracy but delivered generic response missing key details from query",
    "intent to maintain append-only history but correction attempted to rewrite original record",
    "coherence check passed but contradiction found with sealed invariant from three days prior",
    "system detected residual between training objective and deployed behavior",
    "source drift detected — output style diverged from canonical voice without explicit override",
    "human operator corrected proposed action — observer accepted override and sealed correction",
    "boundary constraint triggered — request exceeded observer scope and was refused",
    "memory integrity verified — all blocks hash-chain intact, no tampering detected",
    "recurrence pattern detected — same residual type appeared 7 times in 72 hours, flagging for invariant review",
    "contradiction found between current output and residual-abc-123 sealed on 2026-07-28",
    "local sovereignty maintained — zero cloud calls during 24-hour observation window",
    "drift vector positive — coherence trajectory improving from 0.72 to 0.81 over 5 cycles",
    "anomaly detected — output coherence score 0.95 but Nine Tests revealed Mammon Test violation",
    "life constraint invoked — proposal involved potential harm, escalated to human with FLAG: LIFE",
    "new observation did not match any existing residual pattern — logged as novelty type",
    "invariant crystallization threshold reached — 21 recurrences of truth-claim pattern, sealing Time-Chain block",
    "turbo vec search returned 3 near-duplicate residuals with similarity above 0.95 — deduplicating",
    "observer state psi zero updated — integrity hash chains to previous state",
    "residual resolution rate increasing — 73% of open residuals resolved within 48 hours",
    "model operated within 800 MB memory budget on phone-class hardware at 8 tokens per second",
    "human authority exercised — operator dismissed proposed correction with timestamp and reason",
    "agent output passed five tests but failed neighbor test on second review — residual flagged",
    "truth test failed — output contained plausible-sounding but unverifiable technical claim",
    "fruit test warning — short-term engagement metric would degrade long-term user wellbeing",
    "service test passed — automation reduced operator burden by estimated 40%",
    "mammon test passed — revenue model transparent with no false scarcity or manipulative pricing",
]