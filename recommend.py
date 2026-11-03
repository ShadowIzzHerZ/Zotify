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


def candidates(q, beam=BEAM):
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
    return np.array([i for _, i in items[:POOL]])


def decode(q, k=10, beam=BEAM):
    # rerank a wider pool with exact cosine: the quantised score alone is lossy
    pool = candidates(q, beam)
    return pool[np.argsort(-(emb[pool] @ q))[:k]].tolist()


# --- mock user profile and familiar / non-habitual switch ---
PROFILE_GENRES = ["indie", "pop", "acoustic", "singer-songwriter", "folk", "indie-pop", "chill", "ambient", "piano", "rock", "edm", "hip-hop"]
W = 0.15  # max score shift from the switch, in cosine units


def make_profile(n=120, seed=0):
    pool = cat[cat.track_genre.isin(PROFILE_GENRES)].sort_values("popularity", ascending=False).head(2000)
    return pool.sample(n=n, random_state=seed).item_id.to_numpy()


profile = make_profile()


def prefixes(profile_items):
    return set(ids[profile_items, 0].tolist()), set(map(tuple, ids[profile_items, :2].tolist()))


def familiarity(items, profile_items=None):
    """0..1: half for sharing the level-1 code with the profile, half for level-1+2."""
    P1, P2 = prefixes(profile if profile_items is None else profile_items)
    return np.array([0.5 * (ids[i, 0] in P1) + 0.5 * (tuple(ids[i, :2]) in P2) for i in items])


def recommend(query, k=10, lam=0.0, q=None, profile_items=None):
    """lam in [-1, 1]: +1 favours familiar, -1 favours non-habitual, 0 is neutral."""
    q = embed_query(query) if q is None else q
    pool = candidates(q)
    fam = familiarity(pool, profile_items)
    score = emb[pool] @ q + lam * W * (fam - 0.5) * 2
    order = np.argsort(-score)[:k]
    return pool[order].tolist(), fam[order].tolist()


def show_switch(query, k=10):
    q = embed_query(query)
    print(f"\n> {query}")
    for lam in (-1, 0, 1):
        got, _ = recommend(query, k, lam, q)
        fam = familiarity(got).mean()
        rel = (emb[got] @ q).mean()
        genres = pd.Series(cat.track_genre.iloc[got]).value_counts().head(3).to_dict()
        print(f"  lam={lam:+d}  familiarity {fam:.2f}  mean cosine {rel:.3f}  {genres}")


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
        show_switch(t)
