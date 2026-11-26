"""Ordered linkage reductions without retaining boxed scores for all pairs."""
from contextlib import contextmanager
import mmap
import tempfile

import numpy as np


_BUFFER_BYTES = 8 * 1024**2
_BLOCK_BYTES = 1024**2


@contextmanager
def _score_buffer(size, dtype):
    """Keep small vectors in RAM; spill larger ones to the normal temp directory.

    File-backed pages can become resident during NumPy's reduction, so the
    buffer threshold is not a process RSS limit. Only one cluster pair is
    stored, and the mapping and anonymous temporary file close on exceptions.
    """
    length = size * dtype.itemsize
    if length <= _BUFFER_BYTES:
        yield np.empty(size, dtype=dtype)
    else:
        with tempfile.TemporaryFile() as spool:
            spool.truncate(length)
            with mmap.mmap(spool.fileno(), length, access=mmap.ACCESS_WRITE) as mapping:
                yield np.ndarray(size, dtype=dtype, buffer=mapping)


def aggregate_cluster_matrix(labels, matrix, linkage):
    """Reduce the same ordered scores as labels2clusters, one pair at a time.

    Diagonal groups contain only original diagonal entries. Between different
    groups, scores follow the original i-then-j traversal of the upper triangle,
    with each direction read separately. Contiguous vectors and the unchanged
    NumPy reductions preserve pairwise summation, NaNs and signed zeros.

    Storage is the dense input/output, O(N) membership indices, one contribution
    vector (at most the largest cluster-pair product or diagonal group), and
    gather blocks of at most _BLOCK_BYTES. Large vectors are file-backed.
    """
    count = len(np.unique(labels))
    output = np.zeros((count, count), dtype=np.result_type(matrix.dtype, np.float64))
    if not count:
        return output
    members = [[] for _ in range(count)]
    normalized = np.empty(len(labels), dtype=np.intp)
    for i, label in enumerate(labels):
        members[label].append(i)
        normalized[i] = label % count
    members = [np.asarray(indices, dtype=np.intp) for indices in members]
    sizes = sorted(len(indices) for indices in members)
    size = max(sizes[-1], sizes[-1] * sizes[-2] if count > 1 else 0)
    dtype = matrix.dtype.newbyteorder("=")
    block = max(1, _BLOCK_BYTES // dtype.itemsize)
    reduce = {"average": np.mean, "single": np.min, "complete": np.max}[linkage]
    with _score_buffer(size, dtype) as values:
        for a, left in enumerate(members):
            if len(left):
                for start in range(0, len(left), block):
                    indices = left[start:start + block]
                    values[start:start + len(indices)] = matrix[indices, indices]
                output[a, a] = reduce(values[:len(left)])
            for b, right in enumerate(members):
                if a == b or not len(left) or not len(right):
                    continue
                pos = 0
                for i in np.sort(np.concatenate((left, right))):
                    forward = normalized[i] == a
                    others = right if forward else left
                    first = np.searchsorted(others, i, side="right")
                    for start in range(first, len(others), block):
                        indices = others[start:start + block]
                        values[pos:pos + len(indices)] = matrix[i, indices] if forward else matrix[indices, i]
                        pos += len(indices)
                output[a, b] = reduce(values[:pos])
    return output
