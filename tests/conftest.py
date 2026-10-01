import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import torch  # noqa: E402

try:
    import torch_sparse as upstream  # noqa: E402
    HAS_UPSTREAM = True
except Exception:  # pragma: no cover - upstream is optional
    upstream = None
    HAS_UPSTREAM = False

# Parity targets. Upstream `torch_sparse` is not installable from conda-forge and
# only ships source distributions on PyPI, so the reference for every kernel is
# PyTorch's own implementation of the same algorithm plus SciPy's CSR/CSC
# machinery. `HAS_UPSTREAM` turns on the extra comparisons when a working
# upstream build happens to be available.
DTYPES = [torch.float64, torch.float32]


def rand_coo(rows: int, cols: int, nnz: int, dtype: torch.dtype, seed: int = 0):
    """A random COO index/value pair, deliberately *not* sorted."""
    g = torch.Generator().manual_seed(seed)
    row = torch.randint(0, rows, (nnz, ), generator=g)
    col = torch.randint(0, cols, (nnz, ), generator=g)
    value = torch.randn(nnz, generator=g).to(dtype)
    return row, col, value


def sparse(rows: int, cols: int, nnz: int, dtype: torch.dtype = torch.float64,
           seed: int = 0):
    import mojo_torch_sparse as ts
    row, col, value = rand_coo(rows, cols, nnz, dtype, seed)
    return ts.SparseTensor(row=row, col=col, value=value,
                           sparse_sizes=(rows, cols))


def dense_of(src):
    return src.to_dense()
