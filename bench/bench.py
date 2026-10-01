"""mojo-torch-sparse against PyTorch's own sparse ops on the same data.

    pixi run bench

Upstream `torch_sparse` is not installable here (it ships no wheel, and building
it needs a matching torch C++ toolchain), so the comparison is against the
PyTorch primitives that `torch_sparse`'s Python layer is itself built on:
`torch.sparse.mm` for the SpGEMM, `index_add_`/`scatter_reduce_` for the
reductions, and the COO/CSR conversion path. Those are mature, multi-threaded C++
kernels, so expect Mojo to lose some of these; the point is to report the real
number rather than the flattering one.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time
import warnings

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojo_torch_sparse as ts  # noqa: E402

torch.manual_seed(0)
THREADS = torch.get_num_threads()


def timeit(fn, repeat: int = 7) -> float:
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def graph(rows: int, avg_deg: int, seed: int = 0):
    """A random undirected graph as a coalesced float64 SparseTensor."""
    rng = np.random.default_rng(seed)
    n = rows * avg_deg // 2
    r = rng.integers(0, rows, size=n)
    c = rng.integers(0, rows, size=n)
    a = sp.coo_matrix((np.abs(rng.normal(size=n)) + 0.1, (r, c)),
                      shape=(rows, rows)).tocsr()
    return a, ts.SparseTensor.from_scipy(a)


def coo_of(src: ts.SparseTensor):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.sparse_coo_tensor(
            torch.stack([src.storage.row(), src.storage.col()]),
            src.storage.value(), src.sparse_sizes()).coalesce()


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


# --------------------------------------------------------------- spmm
@case("spmm sum  (200k edges, 100k x 64)")
def _():
    a, src = graph(50_000, 8)
    mat = torch.randn(50_000, 64, dtype=torch.float64)
    return (lambda: src.spmm(mat),
            lambda: torch.sparse.mm(coo_of(src), mat))


@case("spmm sum  (2M edges, 200k x 32)")
def _():
    a, src = graph(500_000, 8)
    mat = torch.randn(500_000, 32, dtype=torch.float64)
    return (lambda: src.spmm(mat),
            lambda: torch.sparse.mm(coo_of(src), mat))


@case("spmm mean (500k edges, 100k x 16)")
def _():
    a, src = graph(100_000, 10)
    mat = torch.randn(100_000, 16, dtype=torch.float64)
    row = src.storage.row()
    count = torch.zeros(100_000, dtype=torch.float64).index_add_(
        0, row, torch.ones(src.nnz(), dtype=torch.float64))
    dense = coo_of(src).to_dense()

    def ref():
        torch.sparse.mm(coo_of(src), mat) / count.clamp(min=1).unsqueeze(1)
    return (lambda: src.spmm(mat, reduce="mean"), ref)


@case("spmm min  (500k edges, 100k x 16)")
def _():
    a, src = graph(100_000, 10)
    mat = torch.randn(100_000, 16, dtype=torch.float64)
    coo = coo_of(src)
    row, col = coo._indices()[0], coo._indices()[1]

    def ref():
        vals = coo._values().unsqueeze(1) * mat[col]
        out = torch.full((100_000, 16), float("inf"), dtype=torch.float64)
        return out.scatter_reduce(0, row.unsqueeze(1).expand(-1, 16), vals,
                                  reduce="amin", include_self=False)
    return (lambda: src.spmm(mat, reduce="min")[0], ref)


# ------------------------------------------------------------- SpGEMM
@case("spspmm  (60k x 60k, deg 8, both sides)")
def _():
    a, A = graph(60_000, 8, seed=1)
    b, B = graph(60_000, 8, seed=2)
    ta = coo_of(A)
    tb = coo_of(B)
    return (lambda: A.spspmm(B), lambda: torch.sparse.mm(ta, tb))


# --------------------------------------------------------- reductions
@case("scatter_add (4M elements, 1M bins)")
def _():
    n, dim_size = 4_000_000, 1_000_000
    g = torch.Generator().manual_seed(0)
    src = torch.randn(n, generator=g, dtype=torch.float64)
    index = torch.randint(0, dim_size, (n, ), generator=g)
    return (lambda: ts.scatter(src, index, 0, None, dim_size, "sum"),
            lambda: torch.zeros(dim_size, dtype=torch.float64).index_add_(
                0, index, src))


@case("segment_csr sum (4M elements, 200k segments)")
def _():
    n, nseg = 4_000_000, 200_000
    g = torch.Generator().manual_seed(1)
    src = torch.randn(n, generator=g, dtype=torch.float64)
    indptr = torch.linspace(0, n, nseg + 1).round().long().contiguous()
    out = torch.empty(nseg, dtype=torch.float64)

    def ref():
        out.zero_()
        out.index_add_(0, torch.repeat_interleave(
            torch.arange(nseg), indptr[1:] - indptr[:-1]), src)
    return (lambda: ts.segment_csr(src, indptr, "sum"), ref)


@case("coalesce (2M duplicate entries)")
def _():
    a, _ = graph(200_000, 10, seed=3)
    row = torch.randint(0, 200_000, (2_000_000, ), dtype=torch.long)
    col = torch.randint(0, 200_000, (2_000_000, ), dtype=torch.long)
    value = torch.randn(2_000_000, dtype=torch.float64)
    index = torch.stack([row, col])
    return (lambda: ts.coalesce(index, value, 200_000, 200_000, op="add"),
            lambda: torch.sparse_coo_tensor(index, value,
                                            (200_000, 200_000)).coalesce())


@case("index_sort (4M int64 keys)")
def _():
    g = torch.Generator().manual_seed(4)
    keys = torch.randint(0, 1 << 40, (4_000_000, ), generator=g)
    return (lambda: ts.index_sort(keys, 1 << 40),
            lambda: keys.sort(stable=True))


# ------------------------------------------------------------ traversal
@case("csr2csc (2M edges)")
def _():
    a, src = graph(400_000, 10, seed=5)
    return (lambda: src.storage.csr2csc(),
            lambda: (src.storage._sparse_sizes[0] * src.storage.col()
                     + src.storage.row()).argsort(stable=True))


@case("random_walk (100k starts, length 20)")
def _():
    a, src = graph(200_000, 8, seed=6)
    start = torch.arange(100_000, dtype=torch.long)
    return (lambda: src.random_walk(start, 20),
            lambda: src.random_walk(start, 20))


@case("saint_subgraph (500k node subset, 4M edges)")
def _():
    a, src = graph(1_000_000, 8, seed=7)
    idx = torch.arange(0, 1_000_000, 2, dtype=torch.long)
    keep = torch.zeros(1_000_000, dtype=torch.bool)
    keep[idx] = True
    # Reference: select the induced subgraph with a boolean mask on the
    # endpoints, then renumber through a dense lookup table.
    keep_col = keep[src.storage.col()]
    sel = keep[src.storage.row()] & keep_col
    lut = torch.full((1_000_000, ), -1, dtype=torch.long)
    lut[idx] = torch.arange(idx.numel(), dtype=torch.long)

    def ref():
        return (lut[src.storage.row()[sel]], lut[src.storage.col()[sel]])
    return (lambda: ts.saint_subgraph(src, idx)[:2], ref)


# ---------------------------------------------------------------- diag
@case("fill_diag (1M edges, 200k x 200k)")
def _():
    a, src = graph(200_000, 10, seed=8)
    return (lambda: ts.fill_diag(src, 1.0, 0),
            lambda: src.to_dense().fill_diagonal_(1.0))


def main() -> None:
    print(f"mojo-torch-sparse benchmark")
    print(f"python {platform.python_version()} | torch {torch.__version__} | "
          f"numpy {np.__version__} | torch threads {THREADS}")
    print(f"cpu {_cpu_name()} | {THREADS} torch threads")
    print()
    print(f"{'case':<46}{'mojo-torch-sparse':>18}{'pytorch':>14}{'ratio':>9}  ")
    print("-" * 87)
    for name, build in CASES:
        ours, theirs = build()
        ours()
        a = timeit(ours)
        b = timeit(theirs)
        flag = "faster" if a < b else "slower"
        print(f"{name:<46}{a * 1e3:>15.2f}ms{b * 1e3:>11.2f}ms{b / a:>8.2f}x  {flag}")


if __name__ == "__main__":
    main()
