"""Trie-constrained beam search over semantic IDs. Query embedded directly (no LLM yet)."""
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from semantic_ids import build_trie

BEAM = 256
POOL = 1000

cat = pd.read_parquet("data/catalog.parquet")
emb = np.load("data/embeddings.npy")
books = np.load("data/codebooks.npy")
ids = np.load("data/semantic_ids.npy")
trie = build_trie(ids)
_model = None


def embed_query(text):
    global _model
    _model = _model or SentenceTransformer("all-MiniLM-L6-v2")
    return _model.encode(text, normalize_embeddings=True)


def decode(q, k=10, beam=BEAM):
    """Beam search level by level; only codes present in the trie are expanded.
    A prefix scores q . (sum of its codewords), the partial reconstruction."""
    beams = [(0.0, np.zeros_like(q), trie)]  # (score, partial vector, trie node)
    for lvl in range(books.shape[0]):
        nxt = []
        for _, vec, node in beams:
            for c, child in node.items():
                v = vec + books[lvl, c]
                nxt.append((float(q @ v), v, child))
        beams = sorted(nxt, key=lambda b: -b[0])[:beam]
    items = [(s, item) for s, _, node in beams for item in node.values()]
    items.sort(key=lambda t: -t[0])
    # rerank a wider pool with exact cosine: the quantised score alone is lossy
    pool = np.array([i for _, i in items[:POOL]])
    return pool[np.argsort(-(emb[pool] @ q))[:k]].tolist()


def show(query, k=10):
    q = embed_query(query)
    got = decode(q, k)
    brute = set(np.argsort(-(emb @ q))[:k].tolist())
    print(f"\n> {query}   (overlap with brute-force top-{k}: {len(brute & set(got))})")
    for i in got:
        r = cat.iloc[i]
        print(f"  {r.track_name[:40]:40} | {r.artists[:25]:25} | {r.track_genre}")


if __name__ == "__main__":
    qs = sys.argv[1:] or ["something calm for a 3 a.m. study session",
                          "high energy workout music", "sad acoustic songs for a rainy day"]
    for t in qs:
        show(t)
