"""mojo-torch-sparse: the sparse-matrix ops of `torch_sparse`, ported to Mojo.

The module layout and the public names mirror upstream
`torch_sparse/__init__.py` (v0.6.18), so `import mojo_torch_sparse as ts` can
stand in for `import torch_sparse as ts` over the covered subset. See the
README's coverage section for what is and is not ported.
"""

__version__ = '0.1.0'

from .storage import SparseStorage  # noqa
from .tensor import SparseTensor  # noqa
from .transpose import t  # noqa
from .narrow import narrow, __narrow_diag__  # noqa
from .select import select  # noqa
from .index_select import index_select, index_select_nnz  # noqa
from .masked_select import masked_select, masked_select_nnz  # noqa
from .permute import permute  # noqa
from .diag import remove_diag, set_diag, fill_diag, get_diag  # noqa
from .add import add, add_, add_nnz, add_nnz_  # noqa
from .mul import mul, mul_, mul_nnz, mul_nnz_  # noqa
from .reduce import sum, mean, min, max  # noqa
from .matmul import matmul, spmm_value_bw  # noqa
from .matmul import spmm_sum, spmm_mean, spmm_min, spmm_max  # noqa
from .matmul import spspmm_sum, spspmm_add, spspmm  # noqa
from .cat import cat  # noqa
from .rw import random_walk  # noqa
from .bandwidth import reverse_cuthill_mckee  # noqa
from .saint import saint_subgraph  # noqa

from .convert import to_torch_sparse, from_torch_sparse  # noqa
from .convert import to_scipy, from_scipy  # noqa
from .coalesce import coalesce  # noqa
from .transpose import transpose  # noqa
from .eye import eye  # noqa
from .spmm import spmm  # noqa
from .spspmm import spspmm  # noqa
from .spadd import spadd  # noqa

from .torch_scatter import gather_csr, scatter, segment_csr  # noqa
from .utils import index_sort  # noqa

__all__ = [
    'SparseStorage',
    'SparseTensor',
    't',
    'narrow',
    '__narrow_diag__',
    'select',
    'index_select',
    'index_select_nnz',
    'masked_select',
    'masked_select_nnz',
    'permute',
    'remove_diag',
    'set_diag',
    'fill_diag',
    'get_diag',
    'add',
    'add_',
    'add_nnz',
    'add_nnz_',
    'mul',
    'mul_',
    'mul_nnz',
    'mul_nnz_',
    'sum',
    'mean',
    'min',
    'max',
    'matmul',
    'cat',
    'random_walk',
    'reverse_cuthill_mckee',
    'saint_subgraph',
    'to_torch_sparse',
    'from_torch_sparse',
    'to_scipy',
    'from_scipy',
    'coalesce',
    'transpose',
    'eye',
    'spmm',
    'spspmm',
    'spadd',
    # Ported `torch_scatter` entry points and `torch_sparse.utils.index_sort`,
    # which upstream reaches through those packages rather than exporting.
    'gather_csr',
    'scatter',
    'segment_csr',
    'index_sort',
    'spmm_value_bw',
    'spmm_sum',
    'spmm_mean',
    'spmm_min',
    'spmm_max',
    'spspmm_sum',
    'spspmm_add',
    '__version__',
]
