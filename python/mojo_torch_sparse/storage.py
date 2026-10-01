"""`torch_sparse/storage.py`."""

import warnings
from typing import Any, List, Optional, Tuple

import torch

from .torch_scatter import scatter, segment_csr
from .utils import Final, addr, index_sort

layouts: Final[List[str]] = ['coo', 'csr', 'csc']


def get_layout(layout: Optional[str] = None) -> str:
    if layout is None:
        layout = 'coo'
        warnings.warn('`layout` argument unset, using default layout '
                      '"coo". This may lead to unexpected behaviour.')
    assert layout == 'coo' or layout == 'csr' or layout == 'csc'
    return layout


class SparseStorage(object):
    _row: Optional[torch.Tensor]
    _rowptr: Optional[torch.Tensor]
    _col: torch.Tensor
    _value: Optional[torch.Tensor]
    _sparse_sizes: Tuple[int, int]
    _rowcount: Optional[torch.Tensor]
    _colptr: Optional[torch.Tensor]
    _colcount: Optional[torch.Tensor]
    _csr2csc: Optional[torch.Tensor]
    _csc2csr: Optional[torch.Tensor]

    def __init__(
        self,
        row: Optional[torch.Tensor] = None,
        rowptr: Optional[torch.Tensor] = None,
        col: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
        rowcount: Optional[torch.Tensor] = None,
        colptr: Optional[torch.Tensor] = None,
        colcount: Optional[torch.Tensor] = None,
        csr2csc: Optional[torch.Tensor] = None,
        csc2csr: Optional[torch.Tensor] = None,
        is_sorted: bool = False,
        trust_data: bool = False,
    ):

        assert row is not None or rowptr is not None
        assert col is not None
        assert col.dtype == torch.long
        assert col.dim() == 1
        col = col.contiguous()

        M: int = 0
        if sparse_sizes is None or sparse_sizes[0] is None:
            if rowptr is not None:
                M = rowptr.numel() - 1
            elif row is not None and row.numel() > 0:
                M = int(row.max()) + 1
        else:
            _M = sparse_sizes[0]
            assert _M is not None
            M = _M
            if rowptr is not None:
                assert rowptr.numel() - 1 == M
            elif row is not None and row.numel() > 0:
                assert trust_data or int(row.max()) < M

        N: int = 0
        if sparse_sizes is None or sparse_sizes[1] is None:
            if col.numel() > 0:
                N = int(col.max()) + 1
        else:
            _N = sparse_sizes[1]
            assert _N is not None
            N = _N
            if col.numel() > 0:
                assert trust_data or int(col.max()) < N

        sparse_sizes = (M, N)

        if row is not None:
            assert row.dtype == torch.long
            assert row.device == col.device
            assert row.dim() == 1
            assert row.numel() == col.numel()
            row = row.contiguous()

        if rowptr is not None:
            assert rowptr.dtype == torch.long
            assert rowptr.device == col.device
            assert rowptr.dim() == 1
            assert rowptr.numel() - 1 == sparse_sizes[0]
            rowptr = rowptr.contiguous()

        if value is not None:
            assert value.device == col.device
            assert value.size(0) == col.size(0)
            value = value.contiguous()

        if rowcount is not None:
            assert rowcount.dtype == torch.long
            assert rowcount.device == col.device
            assert rowcount.dim() == 1
            assert rowcount.numel() == sparse_sizes[0]
            rowcount = rowcount.contiguous()

        if colptr is not None:
            assert colptr.dtype == torch.long
            assert colptr.device == col.device
            assert colptr.dim() == 1
            assert colptr.numel() - 1 == sparse_sizes[1]
            colptr = colptr.contiguous()

        if colcount is not None:
            assert colcount.dtype == torch.long
            assert colcount.device == col.device
            assert colcount.dim() == 1
            assert colcount.numel() == sparse_sizes[1]
            colcount = colcount.contiguous()

        if csr2csc is not None:
            assert csr2csc.dtype == torch.long
            assert csr2csc.device == col.device
            assert csr2csc.dim() == 1
            assert csr2csc.numel() == col.size(0)
            csr2csc = csr2csc.contiguous()

        if csc2csr is not None:
            assert csc2csr.dtype == torch.long
            assert csc2csr.device == col.device
            assert csc2csr.dim() == 1
            assert csc2csr.numel() == col.size(0)
            csc2csr = csc2csr.contiguous()

        self._row = row
        self._rowptr = rowptr
        self._col = col
        self._value = value
        self._sparse_sizes = tuple(sparse_sizes)
        self._rowcount = rowcount
        self._colptr = colptr
        self._colcount = colcount
        self._csr2csc = csr2csc
        self._csc2csr = csc2csr

        if not is_sorted and self._col.numel() > 0:
            idx = self._col.new_zeros(self._col.numel() + 1)
            idx[1:] = self.row()
            idx[1:] *= self._sparse_sizes[1]
            idx[1:] += self._col
            if (idx[1:] < idx[:-1]).any():
                max_value = self._sparse_sizes[0] * self._sparse_sizes[1]
                _, perm = index_sort(idx[1:], max_value)
                self._row = self.row()[perm]
                self._col = self._col[perm]
                if value is not None:
                    self._value = value[perm]
                self._csr2csc = None
                self._csc2csr = None

    @classmethod
    def empty(self):
        row = torch.tensor([], dtype=torch.long)
        col = torch.tensor([], dtype=torch.long)
        return SparseStorage(
            row=row,
            rowptr=None,
            col=col,
            value=None,
            sparse_sizes=(0, 0),
            rowcount=None,
            colptr=None,
            colcount=None,
            csr2csc=None,
            csc2csr=None,
            is_sorted=True,
            trust_data=True,
        )

    def has_row(self) -> bool:
        return self._row is not None

    def row(self):
        from ._lib import call, empty_i64
        row = self._row
        if row is not None:
            return row

        rowptr = self._rowptr
        if rowptr is not None:
            nnz = self._col.numel()
            row = empty_i64(nnz)
            call("mts_ptr2ind", addr(rowptr), addr(row), rowptr.numel(), nnz)
            self._row = row
            return row

        raise ValueError

    def has_rowptr(self) -> bool:
        return self._rowptr is not None

    def rowptr(self) -> torch.Tensor:
        from ._lib import call, empty_i64
        rowptr = self._rowptr
        if rowptr is not None:
            return rowptr

        row = self._row
        if row is not None:
            rowptr = empty_i64(self._sparse_sizes[0] + 1)
            call(
                "mts_ind2ptr", addr(row), addr(rowptr), row.numel(),
                self._sparse_sizes[0],
            )
            self._rowptr = rowptr
            return rowptr

        raise ValueError

    def col(self) -> torch.Tensor:
        return self._col

    def has_value(self) -> bool:
        return self._value is not None

    def value(self) -> Optional[torch.Tensor]:
        return self._value

    def set_value_(
        self,
        value: Optional[torch.Tensor],
        layout: Optional[str] = None,
    ):
        if value is not None:
            if get_layout(layout) == 'csc':
                value = value[self.csc2csr()]
            value = value.contiguous()
            assert value.device == self._col.device
            assert value.size(0) == self._col.numel()

        self._value = value
        return self

    def set_value(
        self,
        value: Optional[torch.Tensor],
        layout: Optional[str] = None,
    ):
        if value is not None:
            if get_layout(layout) == 'csc':
                value = value[self.csc2csr()]
            value = value.contiguous()
            assert value.device == self._col.device
            assert value.size(0) == self._col.numel()

        return SparseStorage(
            row=self._row,
            rowptr=self._rowptr,
            col=self._col,
            value=value,
            sparse_sizes=self._sparse_sizes,
            rowcount=self._rowcount,
            colptr=self._colptr,
            colcount=self._colcount,
            csr2csc=self._csr2csc,
            csc2csr=self._csc2csr,
            is_sorted=True,
            trust_data=True,
        )

    def sparse_sizes(self) -> Tuple[int, int]:
        return self._sparse_sizes

    def sparse_size(self, dim: int) -> int:
        return self._sparse_sizes[dim]

    def sparse_resize(self, sparse_sizes: Tuple[int, int]):
        assert len(sparse_sizes) == 2
        old_sparse_sizes, nnz = self._sparse_sizes, self._col.numel()

        diff_0 = sparse_sizes[0] - old_sparse_sizes[0]
        rowcount, rowptr = self._rowcount, self._rowptr
        if diff_0 > 0:
            if rowptr is not None:
                rowptr = torch.cat([rowptr, rowptr.new_full((diff_0, ), nnz)])
            if rowcount is not None:
                rowcount = torch.cat([rowcount, rowcount.new_zeros(diff_0)])
        elif diff_0 < 0:
            if rowptr is not None:
                rowptr = rowptr[:diff_0]
            if rowcount is not None:
                rowcount = rowcount[:diff_0]

        diff_1 = sparse_sizes[1] - old_sparse_sizes[1]
        colcount, colptr = self._colcount, self._colptr
        if diff_1 > 0:
            if colptr is not None:
                colptr = torch.cat([colptr, colptr.new_full((diff_1, ), nnz)])
            if colcount is not None:
                colcount = torch.cat([colcount, colcount.new_zeros(diff_1)])
        elif diff_1 < 0:
            if colptr is not None:
                colptr = colptr[:diff_1]
            if colcount is not None:
                colcount = colcount[:diff_1]

        return SparseStorage(
            row=self._row,
            rowptr=rowptr,
            col=self._col,
            value=self._value,
            sparse_sizes=sparse_sizes,
            rowcount=rowcount,
            colptr=colptr,
            colcount=colcount,
            csr2csc=self._csr2csc,
            csc2csr=self._csc2csr,
            is_sorted=True,
            trust_data=True,
        )

    def sparse_reshape(self, num_rows: int, num_cols: int):
        assert num_rows > 0 or num_rows == -1
        assert num_cols > 0 or num_cols == -1
        assert num_rows > 0 or num_cols > 0

        total = self.sparse_size(0) * self.sparse_size(1)

        if num_rows == -1:
            num_rows = total // num_cols

        if num_cols == -1:
            num_cols = total // num_rows

        assert num_rows * num_cols == total

        idx = self.sparse_size(1) * self.row() + self.col()

        row = torch.div(idx, num_cols, rounding_mode='floor')
        col = idx % num_cols
        assert row.dtype == torch.long and col.dtype == torch.long

        return SparseStorage(
            row=row,
            rowptr=None,
            col=col,
            value=self._value,
            sparse_sizes=(num_rows, num_cols),
            rowcount=None,
            colptr=None,
            colcount=None,
            csr2csc=None,
            csc2csr=None,
            is_sorted=True,
            trust_data=True,
        )

    def has_rowcount(self) -> bool:
        return self._rowcount is not None

    def rowcount(self) -> torch.Tensor:
        rowcount = self._rowcount
        if rowcount is not None:
            return rowcount

        rowptr = self.rowptr()
        rowcount = rowptr[1:] - rowptr[:-1]
        self._rowcount = rowcount
        return rowcount

    def has_colptr(self) -> bool:
        return self._colptr is not None

    def colptr(self) -> torch.Tensor:
        from ._lib import call, empty_i64
        colptr = self._colptr
        if colptr is not None:
            return colptr

        csr2csc = self._csr2csc
        if csr2csc is not None:
            colptr = empty_i64(self._sparse_sizes[1] + 1)
            # The permuted `col` has to outlive the call: `addr` hands the
            # kernel a bare address, so a temporary would be freed underneath it.
            col_in_csc_order = self._col[csr2csc]
            call(
                "mts_ind2ptr", addr(col_in_csc_order), addr(colptr),
                self._col.numel(), self._sparse_sizes[1],
            )
        else:
            colptr = self._col.new_zeros(self._sparse_sizes[1] + 1)
            torch.cumsum(self.colcount(), dim=0, out=colptr[1:])
        self._colptr = colptr
        return colptr

    def has_colcount(self) -> bool:
        return self._colcount is not None

    def colcount(self) -> torch.Tensor:
        colcount = self._colcount
        if colcount is not None:
            return colcount

        colptr = self._colptr
        if colptr is not None:
            colcount = colptr[1:] - colptr[:-1]
        else:
            colcount = scatter(
                torch.ones_like(self._col), self._col, 0, None,
                self._sparse_sizes[1], 'sum',
            )
        self._colcount = colcount
        return colcount

    def has_csr2csc(self) -> bool:
        return self._csr2csc is not None

    def csr2csc(self) -> torch.Tensor:
        csr2csc = self._csr2csc
        if csr2csc is not None:
            return csr2csc

        idx = self._sparse_sizes[0] * self._col + self.row()
        max_value = self._sparse_sizes[0] * self._sparse_sizes[1]
        _, csr2csc = index_sort(idx, max_value)
        self._csr2csc = csr2csc
        return csr2csc

    def has_csc2csr(self) -> bool:
        return self._csc2csr is not None

    def csc2csr(self) -> torch.Tensor:
        csc2csr = self._csc2csr
        if csc2csr is not None:
            return csc2csr

        max_value = self._sparse_sizes[0] * self._sparse_sizes[1]
        _, csc2csr = index_sort(self.csr2csc(), max_value)
        self._csc2csr = csc2csr
        return csc2csr

    def is_coalesced(self) -> bool:
        idx = self._col.new_full((self._col.numel() + 1, ), -1)
        idx[1:] = self._sparse_sizes[1] * self.row() + self._col
        return bool((idx[1:] > idx[:-1]).all())

    def coalesce(self, reduce: str = "add"):
        idx = self._col.new_full((self._col.numel() + 1, ), -1)
        idx[1:] = self._sparse_sizes[1] * self.row() + self._col
        mask = idx[1:] > idx[:-1]

        if bool(mask.all()):  # Skip if indices are already coalesced.
            return self

        row = self.row()[mask]
        col = self._col[mask]

        value = self._value
        if value is not None:
            ptr = mask.nonzero().flatten()
            ptr = torch.cat([ptr, ptr.new_full((1, ), value.size(0))])
            value = segment_csr(value, ptr, reduce)

        return SparseStorage(
            row=row,
            rowptr=None,
            col=col,
            value=value,
            sparse_sizes=self._sparse_sizes,
            rowcount=None,
            colptr=None,
            colcount=None,
            csr2csc=None,
            csc2csr=None,
            is_sorted=True,
            trust_data=True,
        )

    def fill_cache_(self):
        self.row()
        self.rowptr()

    def clear_cache_(self):
        self._rowcount = None
        self._colptr = None
        self._colcount = None
        self._csr2csc = None
        self._csc2csr = None
        return self

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False

        if self._sparse_sizes != other._sparse_sizes:
            return False
        if self._col.numel() != other._col.numel():
            return False

        rowA, colA, valueA = self.row(), self.col(), self._value
        rowB, colB, valueB = other.row(), other.col(), other._value

        if not torch.equal(rowA, rowB):
            return False
        if not torch.equal(colA, colB):
            return False
        if valueA is None and valueB is None:
            return True
        if valueA is None or valueB is None:
            return False
        return torch.equal(valueA, valueB)

    def clone(self) -> 'SparseStorage':
        out = SparseStorage(
            row=None if self._row is None else self._row.clone(),
            rowptr=None if self._rowptr is None else self._rowptr.clone(),
            col=self._col.clone(),
            value=None if self._value is None else self._value.clone(),
            sparse_sizes=self._sparse_sizes,
            rowcount=None if self._rowcount is None else self._rowcount.clone(),
            colptr=None if self._colptr is None else self._colptr.clone(),
            colcount=None if self._colcount is None else self._colcount.clone(),
            csr2csc=None if self._csr2csc is None else self._csr2csc.clone(),
            csc2csr=None if self._csc2csr is None else self._csc2csr.clone(),
            is_sorted=True,
            trust_data=True,
        )
        return out

    def type(self, dtype: torch.dtype, non_blocking: bool = False):
        value = self._value
        if value is None:
            return self
        return self.set_value(value.to(dtype=dtype, non_blocking=non_blocking))

    def to(self, *args, **kwargs):
        value = self._value
        if value is not None:
            self.set_value_(value.to(*args, **kwargs))
        return self

    def pin_memory(self):
        return self.to('cpu', non_blocking=False)

    def is_pinned(self) -> bool:
        return self._col.is_pinned()

    def __repr__(self) -> str:
        i = ' ' * 9
        row = f'row={self._row}\n{i}'
        rowptr = f'rowptr={self._rowptr}\n{i}'
        rowcount = f'rowcount={self._rowcount}\n{i}'
        col = f'col={self._col}\n{i}'
        colptr = f'colptr={self._colptr}\n{i}'
        colcount = f'colcount={self._colcount}\n{i}'
        value = f'value={self._value}\n{i}'
        csr2csc = f'csr2csc={self._csr2csc}\n{i}'
        csc2csr = f'csc2csr={self._csc2csr}\n{i}'
        sparse_sizes = f'sparse_sizes={self._sparse_sizes}\n{i}'
        return f'SparseStorage(\n{i}{row}{rowptr}{rowcount}{col}{colptr}' \
               f'{colcount}{value}{csr2csc}{csc2csr}{sparse_sizes})'
