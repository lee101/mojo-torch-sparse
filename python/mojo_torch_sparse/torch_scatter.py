"""The `torch_scatter` entry points that `torch_sparse` calls into.

`torch_sparse` depends on `torch_scatter` for `gather_csr`, `segment_csr` and
`scatter`; those three are the ones on its hot paths, so they are ported here
along with the kernels in `src/ported.mojo`. The signatures and the reduction
names are `torch_scatter`'s.
"""

from typing import Optional

import torch

from ._lib import UnsupportedDType, call, empty_i64, scratch_f, tag
from .utils import addr

__all__ = ["gather_csr", "segment_csr", "scatter"]


def _integer_unsupported(op: str, reduce: str) -> UnsupportedDType:
    return UnsupportedDType(
        f"the Mojo kernels are instantiated for float32/float64 only, so the "
        f"integer '{op}' path falls back to torch, which has no integer "
        f"'{reduce}'")


def _integer_scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int,
                     reduce: str) -> torch.Tensor:
    """Integer `scatter`, which no Mojo instantiation covers."""
    if reduce != "sum":
        raise _integer_unsupported("scatter", reduce)
    out = torch.zeros(dim_size, dtype=src.dtype)
    return out.index_add_(0, index, src.reshape(-1))

_REDUCE = ("sum", "mean", "min", "max")

# `torch_scatter` accepts `add` as a spelling of `sum`; the Mojo kernels only
# know the canonical names.
_ALIAS = {"add": "sum", "mul": "sum"}


def _canon(reduce: str) -> str:
    reduce = _ALIAS.get(reduce, reduce)
    assert reduce in _REDUCE, f"unknown reduce '{reduce}'"
    return reduce


def gather_csr(src: torch.Tensor, indptr: torch.Tensor) -> torch.Tensor:
    """Expand `src` so that segment `i` repeats `src[i]` `indptr[i+1]-indptr[i]`
    times."""
    src = src.contiguous()
    indptr = indptr.contiguous()
    if not src.is_floating_point():
        # `index_select` and the `add`/`mul` row-wise path gather int64
        # pointers; there is no Mojo instantiation for those.
        return src.repeat_interleave(indptr[1:] - indptr[:-1])
    out = torch.empty(indptr[-1].item(), dtype=src.dtype)
    call(
        f"mts_gather_csr_{tag(src.dtype)}",
        addr(src), addr(indptr), addr(out), indptr.numel(),
    )
    return out


def segment_csr(src: torch.Tensor, indptr: torch.Tensor,
                reduce: str = "sum") -> torch.Tensor:
    """Reduce `src` over the segments described by `indptr`."""
    return segment_csr_arg(src, indptr, reduce)[0]


def segment_csr_arg(src: torch.Tensor, indptr: torch.Tensor, reduce: str):
    """`segment_csr` plus the `arg` output that `min`/`max` fill in."""
    reduce = _canon(reduce)
    src = src.contiguous()
    indptr = indptr.contiguous()
    nseg = indptr.numel() - 1
    if not src.is_floating_point():
        if reduce != "sum":
            raise _integer_unsupported("segment_csr", reduce)
        out = torch.zeros(nseg, dtype=src.dtype)
        for i in range(nseg):
            seg = src[int(indptr[i]):int(indptr[i + 1])]
            if seg.numel():
                out[i] = seg.sum()
        return out, empty_i64(nseg)
    out = torch.empty(nseg, dtype=src.dtype)
    arg_out = empty_i64(nseg)
    scratch = scratch_f(src.dtype, 1)
    arg_scratch = empty_i64(1)
    call(
        f"mts_segment_csr_{tag(src.dtype)}_{reduce}",
        addr(src), addr(indptr), addr(out), addr(arg_out), nseg,
        addr(scratch), addr(arg_scratch),
    )
    return out, arg_out


def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
            out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = "sum") -> torch.Tensor:
    """Group `src` by `index` into `dim_size` slots of size `dim`."""
    dim = src.dim() + dim if dim < 0 else dim
    assert dim == 0, "the Mojo port covers the dim=0 case"
    reduce = _canon(reduce)
    assert dim_size is not None, "dim_size is required"
    index = index.reshape(-1).contiguous()
    n = index.numel()
    if not src.is_floating_point():
        # The Mojo kernels are instantiated for float32/float64 only. The one
        # integer caller upstream is `SparseStorage.colcount`, which counts
        # `torch.ones_like(col)`, so this path stays a count rather than a
        # general integer scatter.
        return _integer_scatter(src, index, dim_size, reduce)
    trailing = tuple(src.shape[1:])
    stride = 1
    for d in trailing:
        stride *= d
    src = src.reshape(n, stride).contiguous()
    out = torch.empty(dim_size * stride, dtype=src.dtype)
    arg_out = empty_i64(dim_size * stride)
    counts = empty_i64(dim_size)
    call(
        f"mts_scatter_{tag(src.dtype)}_{reduce}",
        addr(src), addr(index), addr(out), addr(arg_out), addr(counts),
        dim_size, n, stride,
    )
    return out.view((dim_size, ) + trailing)
