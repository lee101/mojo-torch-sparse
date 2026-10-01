"""`torch_sparse/rw.py`."""

import torch

from ._spmm import random_walk_kernel
from .tensor import SparseTensor


def random_walk(src: SparseTensor, start: torch.Tensor,
                walk_length: int) -> torch.Tensor:
    rowptr, col, _ = src.csr()
    return random_walk_kernel(rowptr, col, start, walk_length)


SparseTensor.random_walk = random_walk
