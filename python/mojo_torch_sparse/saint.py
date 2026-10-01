"""`torch_sparse/saint.py`."""

from typing import Tuple

import torch

from ._spmm import saint_subgraph_kernel
from .tensor import SparseTensor


def saint_subgraph(src: SparseTensor, node_idx: torch.Tensor
                   ) -> Tuple[SparseTensor, torch.Tensor]:
    row, col, value = src.coo()
    rowptr = src.storage.rowptr()

    out_row, out_col, edge_index = saint_subgraph_kernel(node_idx, rowptr, col)

    if value is not None:
        value = value[edge_index]

    out = SparseTensor(row=out_row, rowptr=None, col=out_col, value=value,
                       sparse_sizes=(node_idx.size(0), node_idx.size(0)),
                       is_sorted=True)

    return out, edge_index


SparseTensor.saint_subgraph = saint_subgraph
