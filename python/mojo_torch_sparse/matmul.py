"""`torch_sparse/matmul.py`."""

from typing import Optional, Tuple

import torch
from torch import Tensor

from ._spmm import spmm_kernel, spmm_value_bw_kernel, spspmm_kernel
from .storage import SparseStorage
from .tensor import SparseTensor
from .utils import addr, index_sort
from .torch_scatter import segment_csr


def spmm_sum(src: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    rowptr, col, value = src.csr()
    if value is not None:
        value = value.to(other.dtype)
    return spmm_kernel(rowptr, col, value, other, "sum")[0]


def spmm_add(src: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    return spmm_sum(src, other)


def spmm_mean(src: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    rowptr, col, value = src.csr()
    if value is not None:
        value = value.to(other.dtype)
    return spmm_kernel(rowptr, col, value, other, "mean")[0]


def spmm_min(src: SparseTensor,
             other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rowptr, col, value = src.csr()
    if value is not None:
        value = value.to(other.dtype)
    out, arg_out = spmm_kernel(rowptr, col, value, other, "min")
    return out, arg_out


def spmm_max(src: SparseTensor,
             other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rowptr, col, value = src.csr()
    if value is not None:
        value = value.to(other.dtype)
    out, arg_out = spmm_kernel(rowptr, col, value, other, "max")
    return out, arg_out


def spmm(src: SparseTensor,
         other: torch.Tensor,
         reduce: str = "sum") -> torch.Tensor:
    if reduce == 'sum' or reduce == 'add':
        return spmm_sum(src, other)
    elif reduce == 'mean':
        return spmm_mean(src, other)
    elif reduce == 'min':
        return spmm_min(src, other)[0]
    elif reduce == 'max':
        return spmm_max(src, other)[0]
    else:
        raise ValueError


def spmm_value_bw(row: torch.Tensor, rowptr: torch.Tensor, col: torch.Tensor,
                  mat: torch.Tensor, grad: torch.Tensor,
                  reduce: str = "sum") -> torch.Tensor:
    """`torch_sparse::spmm_value_bw`: the gradient w.r.t. the sparse values.

    Upstream reaches it only from inside the autograd `Function`; it is
    exposed here because the Mojo port has no autograd graph.
    """
    return spmm_value_bw_kernel(row, rowptr, col, mat, grad, reduce)


def spspmm_sum(src: SparseTensor, other: SparseTensor) -> SparseTensor:
    from ._lib import call
    from ._spmm import spspmm_kernel

    A = src.coalesce()
    B = other.coalesce()

    rowptr_a, col_a, value_a = A.csr()
    rowptr_b, col_b, value_b = B.csr()

    M = A.sparse_size(0)
    K = A.sparse_size(1)
    N = B.sparse_size(1)
    assert B.sparse_size(0) == K, "inner dimensions do not match"

    out_rowptr, out_col, out_value = spspmm_kernel(
        rowptr_a, col_a, value_a, rowptr_b, col_b, value_b, M, N,
    )

    nnz = out_col.numel()
    row = torch.empty(nnz, dtype=torch.long)
    call("mts_ptr2ind", addr(out_rowptr), addr(row), out_rowptr.numel(), nnz)

    # The fill pass emits one candidate per (a, b) pair, unsorted and possibly
    # with duplicates; sort by (row, col) and fold, exactly as
    # `torch.sparse.mm` returns a coalesced product.
    _, perm = index_sort(N * row + out_col, M * N)
    row, out_col = row[perm], out_col[perm]
    out_value = out_value[perm]

    idx = out_col.new_full((nnz + 1, ), -1)
    idx[1:] = N * row + out_col
    mask = idx[1:] > idx[:-1]
    row, out_col = row[mask], out_col[mask]
    ptr = mask.nonzero().flatten()
    ptr = torch.cat([ptr, ptr.new_full((1, ), nnz)])
    out_value = segment_csr(out_value, ptr, "sum")

    if not (src.has_value() or other.has_value()):
        out_value = None

    return SparseTensor(
        row=row,
        col=out_col,
        value=out_value,
        sparse_sizes=(M, N),
        is_sorted=True,
        trust_data=True,
    )


def spspmm_add(src: SparseTensor, other: SparseTensor) -> SparseTensor:
    return spspmm_sum(src, other)


def spspmm(src: SparseTensor,
           other: SparseTensor,
           reduce: str = "sum") -> SparseTensor:
    if reduce == 'sum' or reduce == 'add':
        return spspmm_sum(src, other)
    elif reduce == 'mean' or reduce == 'min' or reduce == 'max':
        raise NotImplementedError
    else:
        raise ValueError


def matmul(src, other, reduce="sum"):
    """Matrix product of a sparse tensor with either another sparse tensor or a
     dense tensor. The sparse tensor represents an adjacency matrix and is
     stored as a list of edges. This method multiplies elements along the rows
     of the adjacency matrix with the column of the other matrix. In regular
     matrix multiplication, the products are then summed together, but this
     method allows us to use other aggregation functions as well.

    Args:
        src (:class:`SparseTensor`): The sparse tensor.
        other (:class:`Tensor` or :class:`SparseTensor`): The second matrix.
        reduce (string, optional): The function to reduce along the rows of
            :obj:`src` and columns of :obj:`other`. Can be :obj:`"sum"`,
            :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`.
            (default: :obj:`"sum"`)

    :rtype: (:class:`Tensor`)
    """
    if isinstance(other, torch.Tensor):
        return spmm(src, other, reduce)
    elif isinstance(other, SparseTensor):
        return spspmm(src, other, reduce)
    raise ValueError


SparseTensor.spmm = lambda self, other, reduce="sum": spmm(self, other, reduce)
SparseTensor.spspmm = lambda self, other, reduce="sum": spspmm(
    self, other, reduce)
SparseTensor.matmul = lambda self, other, reduce="sum": matmul(
    self, other, reduce)
SparseTensor.__matmul__ = lambda self, other: matmul(self, other, 'sum')
