"""Residual k-means semantic IDs: 3 levels x 256 codes plus a collision code."""
import numpy as np

K, LEVELS = 256, 3


def kmeans(x, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        # squared distance without materialising an (n, k, d) tensor
        d = (x**2).sum(1)[:, None] - 2 * x @ c.T + (c**2).sum(1)[None]
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            c[j] = x[m].mean(0) if m.any() else x[rng.integers(len(x))]
    d = (x**2).sum(1)[:, None] - 2 * x @ c.T + (c**2).sum(1)[None]
    return c, d.argmin(1)


def fit(emb):
    """Return codebooks (LEVELS, K, d) and ids (n, LEVELS + 1)."""
    resid, books, codes = emb.copy(), [], []
    for lvl in range(LEVELS):
        c, a = kmeans(resid, K, seed=lvl)
        books.append(c)
        codes.append(a)
        resid -= c[a]
        print(f"level {lvl + 1}: residual MSE {(resid**2).sum(1).mean():.4f}")
    ids = np.stack(codes, 1)
    # 4th code separates items that share the same (c1, c2, c3)
    _, inv = np.unique(ids, axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    rank = np.zeros(len(ids), dtype=int)
    start = np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1]
    for s, e in zip(start, np.r_[start[1:], len(ids)]):
        rank[order[s:e]] = np.arange(e - s)
    return np.stack(books), np.c_[ids, rank]


def build_trie(ids):
    """Nested dict over code tuples; leaves map the last code to item_id."""
    root = {}
    for item, row in enumerate(ids):
        node = root
        for c in row[:-1]:
            node = node.setdefault(int(c), {})
        node[int(row[-1])] = item
    return root


def main():
    emb = np.load("data/embeddings.npy")
    books, ids = fit(emb)
    np.save("data/codebooks.npy", books)
    np.save("data/semantic_ids.npy", ids)
    n = len(ids)
    print(f"items {n}, unique (c1,c2,c3) {len(np.unique(ids[:, :3], axis=0))}, max collision code {ids[:, 3].max()}")
    for lvl in range(LEVELS):
        u = len(np.unique(ids[:, lvl]))
        print(f"level {lvl + 1}: {u}/{K} codes used")
    p = np.bincount(ids[:, 0], minlength=K) / n
    print(f"level-1 largest cluster {p.max():.1%} of items")
    # every item reachable through the trie
    trie = build_trie(ids)
    assert sum(len(l) for a in trie.values() for b in a.values() for l in b.values()) == n


if __name__ == "__main__":
    main()
