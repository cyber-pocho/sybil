"""
Exact cosine top-k, in numpy.

This is the whole search algorithm, and it is deliberately not FAISS. A flat
index — which is what a few thousand past forecasts warrant — *is* a matrix
multiply followed by a partial sort; FAISS earns its keep when the vector count
gets large enough that you are willing to trade exactness for speed (IVF, HNSW,
PQ), and nothing here is anywhere near that. Carrying a compiled dependency to
call the one routine numpy already provides is how `[vectorsearch]` ended up
declaring three packages before a line of it existed.

The scaling limit is worth stating rather than discovering: this reads every
stored vector on every query. At 384 dimensions and float32 that is ~1.5 KB a
row, so 100k memories is a 150 MB read and a 150 MFLOP matmul — a few hundred
milliseconds, and the point at which the answer is pgvector or a real ANN index,
not a bigger matmul.
"""

import numpy as np


def unit(x: np.ndarray) -> np.ndarray:
    """Scale each row to length 1, leaving zero rows alone instead of dividing by zero.

    Normalising is what turns a dot product into a cosine. Done here, at query
    time, rather than at write time: then the stored vectors carry no invariant
    anyone has to remember, and a row written by an older code path cannot
    silently be the one thing in the matrix that is scaled differently.
    """
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def cosine_top_k(matrix: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The `k` rows of `matrix` most similar to `query`, best first.

    Args:
        matrix: (n, d) stored vectors.
        query:  (d,) the vector to compare against.
        k:      how many to return; capped at n, so asking for more is not an error.

    Returns (indices, scores). Scores are cosine similarities in [-1, 1] — 1 is
    the same direction, 0 unrelated, -1 opposite.
    """
    if matrix.size == 0 or k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    # Both sides unit length, so `m @ q` is exactly cos(angle) per row.
    m = unit(np.asarray(matrix, dtype=np.float32))
    q = unit(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]
    scores = m @ q

    k = min(k, scores.shape[0])

    # argpartition is O(n) and gets the right *set* of k without ordering them;
    # a full argsort would be O(n log n) to order rows we are about to discard.
    # Then sort just those k, which is the only part the caller sees.
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    return top, scores[top]
