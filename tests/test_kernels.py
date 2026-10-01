"""Parity tests for the ported kernels, one section per upstream kernel file.

Every assertion compares against an independent implementation: PyTorch's own
op for the same algorithm (`index_add_`, `scatter_reduce_`,
`torch.sparse.mm`, autograd) or SciPy's CSR/CSC machinery. `test_api.py` covers
the Python-level surface built on top of these.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

import mojo_torch_sparse as ts
from conftest import DTYPES, HAS_UPSTREAM, rand_coo, upstream

from mojo_torch_sparse._lib import UnsupportedDType, call
from mojo_torch_sparse.utils import addr


def csr_of(row, col, value, m, n):
    return sp.coo_matrix((value.detach().numpy(), (row.numpy(), col.numpy())),
                         shape=(m, n)).tocsr()


def ref_spmm(row, col, value, m, n, mat, reduce):
    """Row-wise `mat` reduction, written in plain torch from the CSR form."""
    csr = csr_of(row, col, value, m, n)
    a = csr.toarray()
    indptr = csr.indptr
    out = torch.zeros((m, mat.size(-1)), dtype=mat.dtype)
    for i in range(m):
        entries = [(int(j), float(a[i, j])) for j in np.nonzero(a[i])[0]]
        if reduce == "sum":
            out[i] = torch.from_numpy(a[i]).to(mat.dtype) @ mat
            continue
        if reduce == "mean":
            denom = max(int(indptr[i + 1] - indptr[i]), 1)
            out[i] = torch.from_numpy(a[i] / denom).to(mat.dtype) @ mat
            continue
        if not entries:
            continue  # an empty row reduces to 0, as `Reducer::write` does
        pick = entries[0][1] * mat[entries[0][0]]
        for j, v in entries[1:]:
            contrib = v * mat[j]
            pick = (torch.minimum if reduce == "min" else torch.maximum)(
                pick, contrib)
        out[i] = pick
    return out


# ============================================ csrc/cpu/convert_cpu.cpp
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows,cols,nnz", [(1, 1, 0), (5, 5, 0), (17, 3, 40),
                                           (4, 9, 13)])
def test_ind2ptr_and_ptr2ind_match_scipy(rows, cols, nnz, dtype):
    if nnz == 0:
        row = torch.zeros(0, dtype=torch.long)
        col = torch.zeros(0, dtype=torch.long)
    else:
        row, col, _ = rand_coo(rows, cols, nnz, dtype, seed=rows)
    adt = ts.SparseTensor(row=row, col=col, sparse_sizes=(rows, cols))
    # `ind2ptr` counts the *uncoalesced* entries, so SciPy's `tocsr()` (which
    # folds duplicate (row, col) pairs) is not the reference; the row histogram
    # is.
    expected = np.concatenate(
        [[0], np.cumsum(np.bincount(row.numpy(), minlength=rows))])
    rowptr = adt.storage.rowptr()
    assert torch.equal(rowptr, torch.from_numpy(expected.astype(np.int64)))
    assert int(rowptr[-1]) == row.numel()
    if rows * cols > 0:
        # Same count, different fold: the coalesced form is a strict subset.
        csr = sp.coo_matrix((np.ones(row.numel()),
                             (row.numpy(), col.numpy())), shape=(rows, cols)
                             ).tocsr()
        assert int(csr.nnz) <= int(rowptr[-1])

    back = torch.empty(row.numel(), dtype=torch.long)
    call("mts_ptr2ind", addr(rowptr), addr(back), rowptr.numel(), back.numel())
    assert torch.equal(back, adt.storage.row())


def test_ptr2ind_handles_empty_rows():
    ptr = torch.tensor([0, 2, 2, 5], dtype=torch.long)
    out = torch.empty(5, dtype=torch.long)
    call("mts_ptr2ind", addr(ptr), addr(out), ptr.numel(), out.numel())
    assert out.tolist() == [0, 0, 2, 2, 2]


# ================================================== torch_sparse/utils.py
@pytest.mark.parametrize("n", [0, 1, 2, 37, 1000])
def test_index_sort_is_stable_and_matches_torch_sort(n):
    g = torch.Generator().manual_seed(n)
    x = torch.randint(0, 13, (n, ), generator=g)
    got, perm = ts.index_sort(x, 13)
    assert torch.equal(got, x.sort(stable=True).values)
    assert torch.equal(got, x[perm])
    for k in range(13):
        # A stable sort sends equal keys to increasing *source* positions, so
        # the permutation restricted to one key must be increasing.
        picked = perm[got == k]
        if picked.numel() > 1:
            assert (picked.diff() > 0).all()


def test_index_sort_handles_negative_keys():
    x = torch.tensor([5, -3, 5, 0, -3, 7], dtype=torch.long)
    got, perm = ts.index_sort(x, 0)
    assert got.tolist() == [-3, -3, 0, 5, 5, 7]
    assert torch.equal(x[perm], got)


# ==================================================== torch_scatter kernels
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("nseg", [1, 4, 33])
def test_gather_csr_matches_repeat_interleave(dtype, nseg):
    g = torch.Generator().manual_seed(nseg)
    count = torch.randint(0, 5, (nseg, ), generator=g)
    indptr = torch.zeros(nseg + 1, dtype=torch.long)
    torch.cumsum(count, dim=0, out=indptr[1:])
    src = torch.randn(nseg, generator=g).to(dtype)
    got = ts.gather_csr(src, indptr)
    assert got.dtype == dtype
    assert torch.equal(got, src.repeat_interleave(count))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
def test_segment_csr_matches_torch(dtype, reduce):
    n, nseg = 137, 17
    g = torch.Generator().manual_seed(n)
    src = torch.randn(n, generator=g).to(dtype)
    indptr = torch.linspace(0, n, nseg + 1).round().long().contiguous()

    got = ts.segment_csr(src, indptr, reduce)

    ref = torch.zeros(nseg, dtype=dtype)
    for i in range(nseg):
        seg = src[int(indptr[i]):int(indptr[i + 1])]
        if seg.numel() == 0:
            continue  # `Reducer::write` yields 0 for an empty segment
        if reduce == "sum":
            ref[i] = seg.sum()
        elif reduce == "mean":
            ref[i] = seg.mean()
        elif reduce == "min":
            ref[i] = seg.amin()
        else:
            ref[i] = seg.amax()
    assert got.dtype == dtype
    assert torch.allclose(got, ref)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
def test_scatter_matches_torch_scatter_reduce(dtype, reduce):
    n, dim_size = 211, 23
    g = torch.Generator().manual_seed(n)
    src = torch.randn(n, generator=g).to(dtype)
    index = torch.randint(0, dim_size, (n, ), generator=g)

    got = ts.scatter(src, index, 0, None, dim_size, reduce)

    # `include_self` folds the initial zero into a `mean` count, which is not
    # what `Reducer::write` does, so only `sum` keeps it.
    ref = torch.zeros(dim_size, dtype=dtype).scatter_reduce(
        0, index, src, reduce=reduce, include_self=reduce == "sum")
    assert got.dtype == dtype
    assert got.shape == ref.shape
    assert torch.allclose(got, ref)


def test_scatter_2d_trailing_dims():
    n, dim_size, k = 40, 7, 5
    g = torch.Generator().manual_seed(1)
    src = torch.randn(n, k, generator=g, dtype=torch.float64)
    index = torch.randint(0, dim_size, (n, ), generator=g)
    got = ts.scatter(src, index, 0, None, dim_size, "sum")
    ref = torch.zeros(dim_size, k, dtype=torch.float64)
    ref.index_add_(0, index, src)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref)


# ============================================================= spmm_cpu.cpp
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
def test_spmm_matches_reference(dtype, reduce):
    m, n, k = 23, 19, 11
    row, col, value = rand_coo(m, n, 90, dtype, seed=3)
    src = ts.SparseTensor(row=row, col=col, value=value,
                          sparse_sizes=(m, n)).coalesce(reduce="sum")
    mat = torch.randn(n, k, dtype=dtype)

    out = src.spmm(mat, reduce)

    assert out.dtype == dtype
    assert torch.allclose(out, ref_spmm(row, col, value, m, n, mat, reduce),
                          atol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_spmm_sum_matches_scipy_matmul(dtype):
    m, n, k = 40, 30, 17
    row, col, value = rand_coo(m, n, 200, dtype, seed=5)
    src = ts.SparseTensor(row=row, col=col, value=value,
                          sparse_sizes=(m, n)).coalesce(reduce="sum")
    mat = torch.randn(n, k, dtype=dtype)
    out = src.spmm(mat)
    a = torch.from_numpy(csr_of(row, col, value, m, n).toarray())
    assert torch.allclose(out, a @ mat, atol=1e-5)
    if HAS_UPSTREAM:
        assert torch.allclose(out, upstream.matmul(src, mat), atol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_spmm_batched(dtype):
    b, m, n, k = 3, 11, 7, 5
    row, col, value = rand_coo(m, n, 40, dtype, seed=7)
    src = ts.SparseTensor(row=row, col=col, value=value,
                          sparse_sizes=(m, n)).coalesce(reduce="sum")
    mat = torch.randn(b, n, k, dtype=dtype)
    out = src.spmm(mat)
    assert out.shape == (b, m, k)
    for i in range(b):
        assert torch.allclose(out[i], src.spmm(mat[i]), atol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_spmm_without_value(dtype):
    m, n, k = 9, 6, 4
    row, col, _ = rand_coo(m, n, 30, dtype, seed=11)
    src = ts.SparseTensor(row=row, col=col, sparse_sizes=(m, n))
    mat = torch.randn(n, k, dtype=dtype)
    out = src.spmm(mat)
    ones = torch.ones(row.numel(), dtype=dtype)
    assert torch.allclose(out, ref_spmm(row, col, ones, m, n, mat, "sum"),
                          atol=1e-5)


def test_spmm_min_max_arg_selects_the_winning_edge():
    m, n, cols = 7, 5, 3
    row = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.long)
    col = torch.tensor([0, 1, 1, 0, 1, 2], dtype=torch.long)
    value = torch.tensor([1., 2., 3., 4., 5., 6.], dtype=torch.float64)
    src = ts.SparseTensor(row=row, col=col, value=value, sparse_sizes=(m, n))
    mat = torch.randn(n, cols, dtype=torch.float64)

    out_max, arg_max = ts.spmm_max(src, mat)
    out_min, arg_min = ts.spmm_min(src, mat)

    # `arg` is the position in the `col` array, so `value[arg] * mat[col[arg]]`
    # has to reproduce the reduced row.
    # `arg` indexes the storage's own `col`/`value` arrays, which the
    # constructor re-sorted on the way in.
    scol = src.storage.col()
    svalue = src.storage.value()
    k = torch.arange(cols)
    # An empty row reduces to 0 and leaves `arg` untouched, so the identity
    # only holds where the row actually has an entry.
    non_empty = src.storage.rowcount() > 0
    assert torch.allclose(
        svalue[arg_max][non_empty] * mat[scol[arg_max][non_empty], k],
        out_max[non_empty])
    assert torch.allclose(
        svalue[arg_min][non_empty] * mat[scol[arg_min][non_empty], k],
        out_min[non_empty])
    assert (out_max[non_empty] >= out_min[non_empty] - 1e-9).all()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_spmm_value_bw_matches_autograd(dtype, reduce):
    m, n, k = 12, 10, 6
    row, col, value = rand_coo(m, n, 50, dtype, seed=13)
    src = ts.SparseTensor(row=row, col=col, value=value,
                          sparse_sizes=(m, n)).coalesce(reduce="sum")
    mat = torch.randn(n, k, dtype=dtype)
    grad = torch.randn(m, k, dtype=dtype)

    got = ts.spmm_value_bw(src.storage.row(), src.storage.rowptr(),
                           src.storage.col(), mat, grad, reduce)

    v = src.storage.value().clone().requires_grad_(True)
    if reduce == "mean":
        # The kernel divides every edge of a row by that row's `rowcount`,
        # so the reference has to weight the edges the same way before the
        # `index_put` folds them.
        rowcount = src.storage.rowcount().to(dtype)[src.storage.row()].clamp(min=1)
        weighted = v / rowcount
    else:
        weighted = v
    dense = torch.zeros(m, n, dtype=dtype).index_put(
        (src.storage.row(), src.storage.col()), weighted, accumulate=True)
    (dense @ mat).backward(grad)

    assert torch.allclose(got, v.grad, atol=1e-5)


# ============================================================== rw_cpu.cpp
def test_random_walk_matches_reference():
    row = torch.tensor([0, 0, 1, 2, 2, 3], dtype=torch.long)
    col = torch.tensor([1, 2, 0, 1, 3, 3], dtype=torch.long)
    src = ts.SparseTensor(row=row, col=col, sparse_sizes=(4, 4))
    start = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    walk_length = 7
    torch.manual_seed(0)
    got = src.random_walk(start, walk_length)

    rowptr, colr = src.storage.rowptr(), src.storage.col()
    torch.manual_seed(0)
    rand = torch.rand((start.numel(), walk_length))
    ref = torch.full((start.numel(), walk_length + 1), -1, dtype=torch.long)
    for n in range(start.numel()):
        cur = int(start[n])
        ref[n, 0] = cur
        for l in range(walk_length):
            rs, re = int(rowptr[cur]), int(rowptr[cur + 1])
            cur = int(colr[rs + int(rand[n, l] * (re - rs))])
            ref[n, l + 1] = cur

    assert torch.equal(got, ref)
    assert (got[:, 1:] >= 0).all()


# ============================================================= diag_cpu.cpp
@pytest.mark.parametrize("k", [-2, -1, 0, 1, 2])
def test_fill_diag_matches_scipy(k):
    m, n = 8, 7
    row, col, value = rand_coo(m, n, 40, torch.float64, seed=17)
    src = ts.SparseTensor(row=row, col=col, value=value,
                          sparse_sizes=(m, n)).coalesce(reduce="sum")

    out = ts.fill_diag(src, 2.5, k)

    ref = src.to_dense()
    # `set_diag` puts entry i at (i, i + k) for k >= 0 and at (i - k, i) for
    # k < 0, matching `start = -k if k < 0 else 0` upstream.
    num_diag = min(m, n - k) if k >= 0 else min(m + k, n)
    for i in range(num_diag):
        if k >= 0:
            ref[i, i + k] = 2.5
        else:
            ref[i - k, i] = 2.5
    assert torch.allclose(out.to_dense(), ref)


@pytest.mark.parametrize("k", [-1, 0, 1])
def test_remove_diag_and_get_diag(k):
    m = n = 6
    a = sp.random(m, n, density=0.5, random_state=0)
    src = ts.SparseTensor.from_scipy(a)
    assert torch.allclose(ts.get_diag(src),
                          torch.from_numpy(a.diagonal().copy()))
    d = ts.remove_diag(src, k).to_dense()
    for i in range(min(m, n)):
        if 0 <= i + k < n:
            assert d[i, i + k] == 0


def test_set_diag_with_explicit_values():
    src = ts.SparseTensor.from_scipy(sp.eye(4, format="csr"))
    out = ts.set_diag(src, torch.full((4, ), 7.0, dtype=torch.float64), 0)
    assert torch.allclose(out.to_dense(), torch.eye(4, dtype=torch.float64) * 7.0)


# =========================================================== saint_cpu.cpp
def test_saint_subgraph_matches_reference():
    row = torch.tensor([0, 0, 1, 2, 3], dtype=torch.long)
    col = torch.tensor([1, 2, 0, 3, 1], dtype=torch.long)
    value = torch.randn(5, dtype=torch.float64)
    src = ts.SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))
    idx = torch.tensor([0, 2, 3], dtype=torch.long)

    out, edge_index = ts.saint_subgraph(src, idx)

    assoc = {int(node): i for i, node in enumerate(idx)}
    ref_row, ref_col, ref_idx = [], [], []
    rowptr = src.storage.rowptr()
    for v_new, v in enumerate(idx.tolist()):
        for j in range(int(rowptr[v]), int(rowptr[v + 1])):
            w = int(col[j])
            if w in assoc:
                ref_row.append(v_new)
                ref_col.append(assoc[w])
                ref_idx.append(j)

    assert out.storage.row().tolist() == ref_row
    assert out.storage.col().tolist() == ref_col
    assert edge_index.tolist() == ref_idx
    assert torch.equal(value[edge_index], out.storage.value())


# ================================================ SpGEMM behind `spspmm`
@pytest.mark.parametrize("dtype", DTYPES)
def test_spspmm_matches_torch_sparse_mm(dtype):
    m, k, n = 14, 11, 9
    a = sp.random(m, k, density=0.3, random_state=1, format="csr")
    b = sp.random(k, n, density=0.3, random_state=2, format="csr")
    A = ts.SparseTensor.from_scipy(a, has_value=True)
    B = ts.SparseTensor.from_scipy(b, has_value=True)

    out = A.spspmm(B)

    ref = torch.from_numpy(a.toarray()) @ torch.from_numpy(b.toarray())
    assert out.sparse_sizes() == (m, n)
    assert torch.allclose(out.to_dense(), ref.to(torch.float64), atol=1e-4)
    assert out.is_coalesced()
    if HAS_UPSTREAM:
        assert torch.allclose(out.to_dense(),
                              upstream.matmul(A, B).to_dense(), atol=1e-4)


def test_spspmm_matches_torch_sparse_coo_mm():
    m, k, n = 10, 8, 7
    a = sp.random(m, k, density=0.4, random_state=3, format="csr")
    b = sp.random(k, n, density=0.4, random_state=4, format="csr")
    A = ts.SparseTensor.from_scipy(a, has_value=True)
    B = ts.SparseTensor.from_scipy(b, has_value=True)
    ta = torch.sparse_coo_tensor(
        torch.stack([A.storage.row(), A.storage.col()]), A.storage.value(),
        (m, k)).coalesce()
    tb = torch.sparse_coo_tensor(
        torch.stack([B.storage.row(), B.storage.col()]), B.storage.value(),
        (k, n)).coalesce()
    ref = torch.sparse.mm(ta, tb).to_dense()
    got = A.spspmm(B).to_dense().to(ref.dtype)
    assert torch.allclose(got, ref, atol=1e-4)


def test_spspmm_duplicate_entries_are_folded():
    m, k, n = 4, 3, 3
    A = ts.SparseTensor(
        row=torch.tensor([0, 0, 1], dtype=torch.long),
        col=torch.tensor([0, 0, 1], dtype=torch.long),
        value=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        sparse_sizes=(m, k), is_sorted=True)
    B = ts.SparseTensor(
        row=torch.tensor([0, 1], dtype=torch.long),
        col=torch.tensor([0, 2], dtype=torch.long),
        value=torch.tensor([5.0, 7.0], dtype=torch.float64),
        sparse_sizes=(k, n), is_sorted=True)
    out = A.spspmm(B).to_dense()
    ref = torch.from_numpy(sp.csr_matrix(
        (np.array([3.0, 3.0]), ([0, 1], [0, 1])), shape=(m, k)
    ).toarray()) @ torch.from_numpy(sp.csr_matrix(
        (np.array([5.0, 7.0]), ([0, 1], [0, 2])), shape=(k, n)
    ).toarray())
    assert torch.allclose(out, ref.to(torch.float64))


# ============================================================== dtype guards
def test_unsupported_value_dtype_is_rejected():
    src = ts.SparseTensor(row=torch.tensor([0, 1], dtype=torch.long),
                          col=torch.tensor([1, 0], dtype=torch.long),
                          value=torch.tensor([1, 2]), sparse_sizes=(2, 2))
    with pytest.raises(UnsupportedDType):
        src.spmm(torch.ones(2, 3, dtype=torch.int32))
