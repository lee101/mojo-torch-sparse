"""Parity tests for the Python surface, checked against SciPy and dense torch.

`tests/test_kernels.py` covers the Mojo kernels; this file covers the
`SparseStorage`/`SparseTensor` layer and the module-level functions that
upstream exports, and compares the whole thing against the dense or SciPy
result of the same operation.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

import mojo_torch_sparse as ts
from conftest import HAS_UPSTREAM, upstream


def rnd(rows, cols, density=0.4, seed=0, dtype=torch.float64):
    a = sp.random(rows, cols, density=density, random_state=seed, format="csr")
    a.data = np.arange(1, a.nnz + 1, dtype=np.float64)
    return a, ts.SparseTensor.from_scipy(a)


# =============================================================== SparseTensor
def test_from_scipy_round_trips_through_every_layout():
    a, src = rnd(9, 7, seed=1)
    for layout in ("coo", "csr", "csc"):
        b = src.to_scipy(layout=layout)
        assert np.allclose(b.toarray(), a.toarray())
    assert torch.allclose(src.to_dense(),
                          torch.from_numpy(a.toarray()))


def test_from_torch_sparse_coo_and_csr():
    a, src = rnd(6, 5, seed=2)
    coo = src.to_torch_sparse_coo_tensor().coalesce()
    assert torch.equal(ts.SparseTensor.from_torch_sparse_coo_tensor(coo).to_dense(),
                       src.to_dense())
    csr = src.to_torch_sparse_csr_tensor()
    assert torch.equal(ts.SparseTensor.from_torch_sparse_csr_tensor(csr).to_dense(),
                       src.to_dense())
    index, value = ts.from_torch_sparse(coo)
    assert torch.equal(index, torch.stack([src.storage.row(), src.storage.col()]))
    assert np.allclose(ts.to_scipy(index, value, 6, 5).toarray(), a.toarray())


def test_from_dense_and_eye():
    mat = torch.tensor([[0.0, 2.0], [3.0, 0.0]])
    src = ts.SparseTensor.from_dense(mat)
    assert torch.equal(src.to_dense(), mat)
    assert torch.equal(ts.SparseTensor.eye(3).to_dense(), torch.eye(3))
    assert torch.equal(ts.SparseTensor.eye(3, 5).to_dense(), torch.eye(3, 5))
    index, value = ts.eye(4)
    assert index.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert value.tolist() == [1.0] * 4


def test_sparse_sizes_nnz_density_and_metadata():
    a, src = rnd(10, 8, density=0.3, seed=3)
    assert src.sparse_sizes() == (10, 8)
    assert src.sizes() == [10, 8]
    assert src.dim() == 2
    assert src.nnz() == a.nnz
    assert src.numel() == a.nnz
    assert src.is_quadratic() is False
    square, _ = rnd(6, 6, density=0.5, seed=4)
    assert ts.SparseTensor.from_scipy(square).is_quadratic() is True
    assert abs(src.density() - a.nnz / 80) < 1e-12
    assert abs(src.sparsity() - (1 - a.nnz / 80)) < 1e-12


def test_equality_and_clone():
    a, src = rnd(7, 7, seed=5)
    other = src.clone()
    assert src == other
    assert src is not other
    assert src.storage is not other.storage
    other.set_value_(other.storage.value() + 1.0)
    assert src != other


def test_unsorted_input_is_sorted_by_the_constructor():
    row = torch.tensor([2, 0, 1, 0], dtype=torch.long)
    col = torch.tensor([1, 2, 0, 0], dtype=torch.long)
    value = torch.tensor([4.0, 5.0, 6.0, 7.0])
    src = ts.SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
    assert src.storage.row().tolist() == [0, 0, 1, 2]
    assert src.storage.col().tolist() == [0, 2, 0, 1]
    assert src.storage.value().tolist() == [7.0, 5.0, 6.0, 4.0]
    assert src.is_coalesced()


def test_sparse_resize_and_reshape():
    a, src = rnd(6, 4, density=0.5, seed=6)
    grown = src.sparse_resize((8, 6))
    assert grown.sparse_sizes() == (8, 6)
    assert grown.storage.rowcount().tolist() == np.bincount(
        src.storage.row().numpy(), minlength=8).tolist()
    # `sparse_reshape` re-indexes `sparse_size(1) * row + col`, so the entries
    # land in row-major order of the flattened matrix, not the original one.
    flat = src.sparse_reshape(3, 8)
    assert flat.sparse_sizes() == (3, 8)
    expected = a.toarray().reshape(-1).reshape(3, 8)
    assert np.allclose(flat.to_dense().numpy(), expected)


def test_is_symmetric_and_to_symmetric():
    a = sp.random(8, 8, density=0.3, random_state=7, format="csr")
    a.data = np.arange(1, a.nnz + 1, dtype=np.float64)
    sym = (a + a.T).tocsr()
    assert ts.SparseTensor.from_scipy(sym).is_symmetric()
    asym = ts.SparseTensor.from_scipy(a)
    assert not asym.is_symmetric()
    # `to_symmetric` mirrors every entry and then folds the pairs, so it is
    # `A + A^T`; applying it twice would double-count.
    assert np.allclose(asym.to_symmetric().to_dense().numpy(),
                       a.toarray() + a.toarray().T)


# ================================================================== coalesce
def test_coalesce_sums_duplicates():
    index = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 1]], dtype=torch.long)
    value = torch.tensor([1.0, 2.0, 3.0, 4.0])
    idx, val = ts.coalesce(index, value, m=2, n=2, op="add")
    # `SparseStorage` sorts by `n * row + col` first, so the result comes back
    # in (0, 1) then (1, 0) order with the three (0, 1) entries folded together.
    assert idx.tolist() == [[0, 1], [1, 0]]
    assert torch.allclose(val, torch.tensor([7.0, 3.0]))
    ref = sp.coo_matrix((value.numpy(), (index[0].numpy(), index[1].numpy())),
                        shape=(2, 2)).tocsr()
    got = sp.coo_matrix((val.numpy(), (idx[0].numpy(), idx[1].numpy())),
                        shape=(2, 2))
    assert np.allclose(got.toarray(), ref.toarray())


def test_spadd_matches_scipy_addition():
    a, A = rnd(7, 7, density=0.3, seed=8)
    b, B = rnd(7, 7, density=0.3, seed=9)
    idx, val = ts.spadd(torch.stack([A.storage.row(), A.storage.col()]),
                        A.storage.value(), torch.stack([B.storage.row(),
                                                        B.storage.col()]),
                        B.storage.value(), 7, 7)
    ref = (a + b).tocsr()
    got = sp.coo_matrix((val.numpy(), (idx[0].numpy(), idx[1].numpy())),
                        shape=(7, 7))
    assert np.allclose(got.toarray(), ref.toarray())


# ================================================================= transpose
def test_transpose_and_t_match_scipy():
    a, src = rnd(6, 4, density=0.4, seed=10)
    assert np.allclose(src.t().to_dense().numpy(), a.toarray().T)
    index, value = ts.transpose(torch.stack([src.storage.row(),
                                             src.storage.col()]),
                                src.storage.value(), 6, 4)
    got = sp.coo_matrix((value.numpy(), (index[0].numpy(), index[1].numpy())),
                        shape=(4, 6))
    assert np.allclose(got.toarray(), a.toarray().T)
    uncoalesced = ts.transpose(torch.stack([src.storage.row(), src.storage.col()]),
                               src.storage.value(), 6, 4, coalesced=False)
    assert uncoalesced[0].shape[1] == src.nnz()


# ====================================================================== cat
def test_cat_dim0_and_dim1():
    a, A = rnd(3, 5, density=0.5, seed=11)
    b, B = rnd(4, 5, density=0.5, seed=12)
    out = ts.cat([A, B], dim=0)
    assert out.sparse_sizes() == (7, 5)
    assert np.allclose(out.to_dense().numpy(),
                       np.vstack([a.toarray(), b.toarray()]))

    c, C = rnd(3, 2, density=0.5, seed=13)
    d, D = rnd(3, 3, density=0.5, seed=14)
    out = ts.cat([C, D], dim=1)
    assert out.sparse_sizes() == (3, 5)
    assert np.allclose(out.to_dense().numpy(),
                       np.hstack([c.toarray(), d.toarray()]))


def test_cat_diag():
    a, A = rnd(3, 3, density=0.5, seed=15)
    b, B = rnd(4, 4, density=0.5, seed=16)
    out = ts.cat([A, B], dim=(0, 1))
    assert out.sparse_sizes() == (7, 7)
    ref = np.zeros((7, 7))
    ref[:3, :3] = a.toarray()
    ref[3:, 3:] = b.toarray()
    assert np.allclose(out.to_dense().numpy(), ref)
    back = ts.__narrow_diag__(out, (3, 3), (4, 4))
    assert np.allclose(back.to_dense().numpy(), b.toarray())


# ================================================== narrow / select / index
def test_narrow_and_select():
    a, src = rnd(7, 6, density=0.4, seed=17)
    assert np.allclose(ts.narrow(src, 0, 2, 3).to_dense().numpy(),
                       a.toarray()[2:5])
    assert np.allclose(ts.narrow(src, 1, 1, 3).to_dense().numpy(),
                       a.toarray()[:, 1:4])
    assert np.allclose(ts.select(src, 0, 4).to_dense().numpy(),
                       a.toarray()[4:5])
    assert np.allclose(ts.select(src, 1, 2).to_dense().numpy(),
                       a.toarray()[:, 2:3])


def test_index_select_and_masked_select():
    a, src = rnd(6, 6, density=0.5, seed=18)
    idx = torch.tensor([0, 3, 5], dtype=torch.long)
    assert np.allclose(ts.index_select(src, 0, idx).to_dense().numpy(),
                       a.toarray()[idx.numpy()])
    assert np.allclose(ts.index_select(src, 1, idx).to_dense().numpy(),
                       a.toarray()[:, idx.numpy()])
    keep = torch.tensor([0, 1], dtype=torch.long)
    picked = ts.index_select_nnz(src, keep)
    assert picked.nnz() == 2
    row, col, _ = picked.coo()
    expected = np.zeros((6, 6))
    expected[row.numpy(), col.numpy()] = a.toarray()[row.numpy(), col.numpy()]
    assert np.allclose(picked.to_dense().numpy(), expected)

    mask = torch.tensor([True, False, True, False, True, False])
    assert np.allclose(ts.masked_select(src, 0, mask).to_dense().numpy(),
                       a.toarray()[mask.numpy()])
    assert np.allclose(ts.masked_select(src, 1, mask).to_dense().numpy(),
                       a.toarray()[:, mask.numpy()])
    nnz_mask = torch.zeros(src.nnz(), dtype=torch.bool)
    nnz_mask[:2] = True
    assert src.masked_select_nnz(nnz_mask).nnz() == 2


def test_permutate():
    a, src = rnd(5, 5, density=0.5, seed=19)
    perm = torch.tensor([4, 3, 2, 1, 0], dtype=torch.long)
    out = ts.permute(src, perm)
    expected = a.toarray()[np.ix_(perm.numpy(), perm.numpy())]
    assert np.allclose(out.to_dense().numpy(), expected)


# ================================================================ add / mul
def test_add_and_mul_with_dense_vectors():
    a, src = rnd(4, 5, density=0.5, seed=20)
    row_vec = torch.arange(1.0, 5.0).reshape(4, 1)
    col_vec = torch.arange(1.0, 6.0).reshape(1, 5)
    dense = a.toarray()
    # Upstream writes the result through `set_value` over the *existing* nnz,
    # so positions that are structurally zero stay zero.
    def expect(other):
        out = dense + other
        return np.where(dense == 0, 0.0, out)

    assert np.allclose(ts.add(src, row_vec).to_dense().numpy(),
                       expect(row_vec.numpy()))
    assert np.allclose(ts.add(src, col_vec).to_dense().numpy(),
                       expect(col_vec.numpy()))
    assert np.allclose(ts.mul(src, col_vec).to_dense().numpy(),
                       np.where(dense == 0, 0.0, dense * col_vec.numpy()))
    assert np.allclose(src.add(row_vec).to_dense().numpy(),
                       expect(row_vec.numpy()))
    assert np.allclose((src * col_vec).to_dense().numpy(),
                       np.where(dense == 0, 0.0, dense * col_vec.numpy()))


def test_add_and_mul_of_sparse_tensors():
    a, A = rnd(5, 5, density=0.4, seed=21)
    b, B = rnd(5, 5, density=0.4, seed=22)
    assert np.allclose(ts.add(A, B).to_dense().numpy(),
                       a.toarray() + b.toarray())
    # `mul` of two coalesced sparse tensors is the elementwise product.
    assert np.allclose(ts.mul(A, B).to_dense().numpy(),
                       a.toarray() * b.toarray())


def test_add_nnz_and_mul_nnz_are_elementwise_on_values():
    _, A = rnd(4, 4, density=0.5, seed=23)
    other = torch.randn(A.nnz(), dtype=torch.float64)
    assert torch.allclose(ts.add_nnz(A, other).storage.value(),
                          A.storage.value() + other)
    assert torch.allclose(ts.mul_nnz(A, other).storage.value(),
                          A.storage.value() * other)


# =========================================================== sum / mean / ...
def test_reductions_match_dense():
    a, src = rnd(6, 7, density=0.4, seed=24)
    dense = torch.from_numpy(a.toarray())
    value = src.storage.value()

    def reduce_over(which, how):
        """`which` picks the row or column index; only the stored values take
        part, exactly as `scatter` / `segment_csr` see them."""
        group = src.storage.col() if which == 0 else src.storage.row()
        size = src.size(1 - which)
        if how == "sum":
            return torch.zeros(size, dtype=dense.dtype).index_add_(0, group, value)
        if how == "max":
            return torch.full((size, ), float("-inf"), dtype=dense.dtype
                              ).scatter_reduce(0, group, value, reduce="amax",
                                               include_self=False)
        if how == "mean":
            count = torch.zeros(size, dtype=dense.dtype).index_add_(
                0, group, torch.ones_like(value))
            total = torch.zeros(size, dtype=dense.dtype).index_add_(0, group, value)
            return total / count.clamp(min=1)
        pick = torch.full((size, ), float("inf"), dtype=dense.dtype)
        pick = pick.scatter_reduce(0, group, value, reduce="amin",
                                   include_self=False)
        return pick

    # The whole-tensor reductions are over `value` directly.
    assert torch.allclose(ts.sum(src), value.sum())
    assert torch.allclose(ts.mean(src), value.mean())
    assert torch.allclose(ts.min(src), value.min())
    assert torch.allclose(ts.max(src), value.max())
    for dim, which in ((0, 0), (1, 1)):
        for how in ("sum", "mean", "min", "max"):
            got = getattr(ts, how)(src, dim=dim)
            ref = reduce_over(which, how)
            assert torch.allclose(got, ref), (how, dim)
    # `sum` agrees with the dense sum because the structural zeros add nothing.
    assert torch.allclose(ts.sum(src, dim=0), dense.sum(dim=0))
    assert torch.allclose(ts.sum(src, dim=1), dense.sum(dim=1))


def test_reductions_without_value():
    idx = torch.tensor([[0, 1, 1], [0, 0, 2]], dtype=torch.long)
    src = ts.SparseTensor(row=idx[0], col=idx[1], sparse_sizes=(3, 3))
    assert ts.sum(src, dim=0).tolist() == [2.0, 0.0, 1.0]
    assert ts.sum(src, dim=1).tolist() == [1.0, 2.0, 0.0]
    assert float(ts.sum(src)) == 3.0


# ==================================================================== matmul
def test_matmul_dispatch_matches_dense():
    a, A = rnd(5, 6, density=0.4, seed=25)
    b, B = rnd(6, 4, density=0.4, seed=26)
    x = torch.randn(6, 3, dtype=torch.float64)
    assert torch.allclose(A.matmul(x), torch.from_numpy(a.toarray()) @ x,
                          atol=1e-5)
    assert torch.allclose(A @ B.to_dense(),
                          torch.from_numpy(a.toarray()) @ b.toarray(), atol=1e-5)
    assert torch.allclose(ts.matmul(A, x), A.spmm(x), atol=1e-5)
    if HAS_UPSTREAM:
        assert torch.allclose(A.matmul(x), upstream.matmul(A, x), atol=1e-5)


def test_coo_spmm_matches_dense():
    row = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
    col = torch.tensor([0, 2, 1, 0, 2], dtype=torch.long)
    value = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    x = torch.randn(3, 4, dtype=torch.float64)
    got = ts.spmm(torch.stack([row, col]), value, 3, 3, x)
    dense = torch.zeros(3, 3, dtype=torch.float64)
    dense[row, col] = value
    assert torch.allclose(got, dense @ x, atol=1e-5)


def test_spspmm_module_function():
    a, A = rnd(4, 5, density=0.5, seed=27)
    b, B = rnd(5, 6, density=0.5, seed=28)
    index, value = ts.spspmm(torch.stack([A.storage.row(), A.storage.col()]),
                             A.storage.value(),
                             torch.stack([B.storage.row(), B.storage.col()]),
                             B.storage.value(), 4, 5, 6)
    got = sp.coo_matrix((value.numpy(), (index[0].numpy(), index[1].numpy())),
                        shape=(4, 6))
    assert np.allclose(got.toarray(), a.toarray() @ b.toarray(), atol=1e-5)


# ==================================================================== random
def test_random_walk_stays_on_edges():
    a, src = rnd(6, 6, density=0.4, seed=29)
    walk = src.random_walk(torch.arange(6, dtype=torch.long), 5)
    assert walk.shape == (6, 6)
    for n in range(6):
        for l in range(1, 6):
            u, v = int(walk[n, l - 1]), int(walk[n, l])
            assert a[u, v] != 0


def test_reverse_cuthill_mckee_permutes_rows_and_columns():
    a, src = rnd(7, 7, density=0.5, seed=30)
    out, perm = src.reverse_cuthill_mckee(is_symmetric=False)
    assert sorted(perm.tolist()) == list(range(7))
    # `is_symmetric=False` makes upstream symmetrize first, so the reference
    # is the permuted `A + A^T`, not the permuted `A`.
    base = a.toarray() + a.toarray().T
    expected = base[np.ix_(perm.numpy(), perm.numpy())]
    assert np.allclose(out.to_dense().numpy(), expected)


# ================================================================ storage API
def test_storage_derived_views_agree_with_scipy():
    a, src = rnd(7, 9, density=0.4, seed=31)
    storage = src.storage
    csr = sp.csr_matrix((a.data, a.indices, a.indptr), shape=(7, 9))
    csc = a.tocsc()
    assert np.array_equal(storage.rowptr().numpy(), csr.indptr)
    assert np.array_equal(storage.rowcount().numpy(), np.diff(csr.indptr))
    assert np.array_equal(storage.colptr().numpy(), csc.indptr)
    assert np.array_equal(storage.colcount().numpy(), np.diff(csc.indptr))
    assert storage.has_row() and storage.has_rowptr() and storage.has_value()
    # csr2csc is a permutation that turns CSR order into CSC order.
    perm = storage.csr2csc()
    assert sorted(perm.tolist()) == list(range(a.nnz))
    assert np.array_equal(storage.row()[perm].numpy(), csc.indices)
    assert (np.diff(storage.col()[perm].numpy()) >= 0).all()
    assert np.array_equal(storage.csc2csr().numpy(), np.argsort(perm.numpy()))


def test_cache_clearing_is_transparent():
    _, src = rnd(5, 5, density=0.5, seed=32)
    before = src.to_dense()
    src.fill_cache_().clear_cache_()
    assert torch.equal(src.to_dense(), before)
