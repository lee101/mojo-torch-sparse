"""`torch_sparse/utils.py`."""

from typing import Any, Optional, Tuple

import torch

try:
    from typing_extensions import Final  # noqa
except ImportError:
    Final = Any


def index_sort(
        inputs: torch.Tensor,
        max_value: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""See pyg-lib documentation for more details:
    https://pyg-lib.readthedocs.io/en/latest/modules/ops.html

    Upstream forwards this to `pyg_lib.ops.index_sort`, a counting sort over
    `max_value`. The Mojo port uses a stable LSD radix sort instead: a stable
    sort by the same key yields the same permutation, and the radix sort does
    not need `max_value` to be small.
    """
    from ._lib import call, empty_i64

    n = inputs.numel()
    # The radix sort alternates between its two buffer pairs, so it writes into
    # the input on every other pass. Upstream never mutates its input, so the
    # kernel gets a private copy.
    keys = inputs.contiguous().to(torch.long).clone()
    payload = empty_i64(n)[:n]
    payload.copy_(torch.arange(n, dtype=torch.long))

    sorted_keys = torch.empty_like(keys)
    # `empty_i64(0)` still has to hand ctypes a non-null address, so the
    # permutation is sliced back to the real length.
    sorted_payload = empty_i64(n)[:n]
    counts = empty_i64(256)

    call(
        "mts_radix_sort",
        addr(keys), addr(payload), addr(sorted_keys), addr(sorted_payload),
        addr(counts), n,
    )
    return sorted_keys, sorted_payload


def is_scalar(other: Any) -> bool:
    return isinstance(other, int) or isinstance(other, float)


def addr(a: torch.Tensor) -> int:
    """Address of an already-contiguous tensor; callers guarantee that."""
    return a.data_ptr()
