"""The Mojo `spmm_cpu` kernel, called the way `csrc/cpu/spmm_cpu.cpp` is.

`src/ported.mojo` keeps upstream's loop; this module only supplies the buffers
upstream allocates (`out`, `arg_out`, `vals`, `args`) and the `Int` addresses
the C ABI needs.
"""

from typing import Optional, Tuple

import torch

from ._lib import call, empty_f32, empty_f64, empty_i64, scratch_f, tag
from .utils import addr

__all__ = ["spmm_kernel", "spmm_value_bw_kernel", "non_diag_mask_kernel",
           "random_walk_kernel", "saint_subgraph_kernel", "spspmm_kernel"]


def spmm_kernel(rowptr: torch.Tensor, col: torch.Tensor,
                value: Optional[torch.Tensor], mat: torch.Tensor,
                reduce: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """`spmm_cpu(rowptr, col, optional_value, mat, reduce)`."""
    assert mat.dim() >= 2
    mat = mat.contiguous()

    N = mat.size(-2)
    K = mat.size(-1)
    B = mat.numel() // (N * K)
    M = rowptr.numel() - 1

    has_value = value is not None
    dtype = value.dtype if has_value else mat.dtype
    t = tag(dtype)

    out = torch.empty((B, M, K), dtype=dtype)
    arg_out = empty_i64(out.numel())
    vals = scratch_f(dtype, K)
    args = empty_i64(K)
    # Pointers are non-nullable, so an absent value still needs a real address.
    value_or_dummy = value if has_value else torch.ones(1, dtype=dtype)

    call(
        f"mts_spmm_{t}{'' if has_value else '_nv'}_{reduce}",
        addr(rowptr), addr(col), addr(value_or_dummy), addr(mat), addr(out),
        addr(arg_out), addr(vals), addr(args), M, N, K, B,
    )

    sizes = list(mat.shape)
    sizes[mat.dim() - 2] = M
    out = out.view(sizes)

    if reduce in ('min', 'max'):
        return out, arg_out.view(sizes)
    return out, None


def spmm_value_bw_kernel(row: torch.Tensor, rowptr: torch.Tensor,
                         col: torch.Tensor, mat: torch.Tensor,
                         grad: torch.Tensor, reduce: str) -> torch.Tensor:
    """`spmm_value_bw_cpu(row, rowptr, col, mat, grad, reduce)`."""
    mat = mat.contiguous()
    grad = grad.contiguous()

    M = grad.size(-2)
    N = mat.size(-2)
    K = mat.size(-1)
    E = row.numel()
    B = mat.numel() // (N * K)

    out = torch.zeros(E, dtype=mat.dtype)
    call(
        f"mts_spmm_value_bw_{tag(mat.dtype)}_{reduce}",
        addr(row), addr(rowptr), addr(col), addr(mat), addr(grad), addr(out),
        M, N, E, K, B,
    )
    return out


def non_diag_mask_kernel(row: torch.Tensor, col: torch.Tensor, M: int,
                          N: int, k: int) -> torch.Tensor:
    """`non_diag_mask_cpu(row, col, M, N, k)`."""
    E = row.numel()
    if k < 0:
        num_diag = min(M + k, N)
    else:
        num_diag = min(M, N - k)
    mask = torch.zeros(E + num_diag, dtype=torch.int8)
    call(
        "mts_non_diag_mask", addr(row), addr(col), addr(mask), E, M, N, k,
    )
    return mask


def random_walk_kernel(rowptr: torch.Tensor, col: torch.Tensor,
                       start: torch.Tensor, walk_length: int) -> torch.Tensor:
    """`random_walk_cpu(rowptr, col, start, walk_length)`.

    The upstream kernel draws its own uniform sample; here Python draws it so
    that the result is reproducible and seedable by the caller.
    """
    batch = start.size(0)
    L = walk_length + 1
    rand = torch.rand((batch, walk_length), dtype=torch.float32)
    out = torch.full((batch, L), -1, dtype=torch.long)
    call(
        "mts_random_walk", addr(rowptr), addr(col), addr(start), addr(rand),
        addr(out), batch, walk_length,
    )
    return out


def saint_subgraph_kernel(idx: torch.Tensor, rowptr: torch.Tensor,
                          col: torch.Tensor):
    """`subgraph_cpu(idx, rowptr, row, col)`.

    Upstream grows three `std::vector`s and reports the length at the end; the
    ABI cannot allocate, so the traversal is run once to count and once to
    fill.
    """
    M = rowptr.size(0) - 1
    assoc = torch.full((M, ), -1, dtype=torch.long)
    assoc[idx] = torch.arange(idx.size(0), dtype=torch.long)

    length = empty_i64(1)
    call(
        "mts_saint_subgraph_count", addr(idx), addr(rowptr), addr(col),
        addr(assoc), addr(length), idx.size(0),
    )
    n = int(length.item())

    row = torch.empty(n, dtype=torch.long)
    out_col = torch.empty(n, dtype=torch.long)
    out_idx = torch.empty(n, dtype=torch.long)
    call(
        "mts_saint_subgraph_fill", addr(idx), addr(rowptr), addr(col),
        addr(assoc), addr(row), addr(out_col), addr(out_idx), idx.size(0),
    )
    return row, out_col, out_idx


def spspmm_kernel(rowptr_a: torch.Tensor, col_a: torch.Tensor,
                  value_a: Optional[torch.Tensor], rowptr_b: torch.Tensor,
                  col_b: torch.Tensor, value_b: Optional[torch.Tensor],
                  M: int, N: int):
    """The SpGEMM behind `SparseTensor.spspmm` (upstream calls
    `torch.sparse.mm`)."""
    has_value = value_a is not None and value_b is not None
    dtype = value_a.dtype if has_value else torch.float64
    t = tag(dtype)

    count = torch.empty(M, dtype=torch.long)
    call("mts_spspmm_count", addr(rowptr_a), addr(col_a), addr(rowptr_b),
         addr(count), M)

    out_rowptr = torch.zeros(M + 1, dtype=torch.long)
    torch.cumsum(count, dim=0, out=out_rowptr[1:])
    nnz = int(out_rowptr[-1].item())

    out_col = torch.empty(nnz, dtype=torch.long)
    out_value = torch.empty(nnz, dtype=dtype)

    va = value_a if value_a is not None else torch.zeros(1, dtype=dtype)
    vb = value_b if value_b is not None else torch.zeros(1, dtype=dtype)
    call(
        f"mts_spspmm_fill_{t}{'' if has_value else '_nv'}",
        addr(rowptr_a), addr(col_a), addr(va), addr(rowptr_b), addr(col_b),
        addr(vb), addr(out_rowptr), addr(out_col), addr(out_value), M,
    )

    out_rowptr[M] = nnz
    return out_rowptr, out_col, out_value
