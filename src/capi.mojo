"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address, because `@export` refuses parametric
functions: the kernels in `ported.mojo` stay parametric (over the element type
and the reduction) and this file instantiates them concretely. Nothing here
allocates, so nothing here can leak; the caller owns the outputs and the
scratch buffers, exactly as upstream owns its `std::vector` scratch.

Argument order is the kernel's, with the destination and the scratch appended
where the ABI needs them spelled out. `M`, `N`, `K` and `B` are spelled out as
`num_rows`, `num_inner`, `num_cols` and `batch` because those single letters
are reserved in this dialect.
"""

from ported import (
    DIV,
    MAX,
    MEAN,
    MIN,
    MUL,
    SUM,
    CPtr,
    F32CPtr,
    IPtr,
    Ptr,
    BPtr,
    gather_csr_cpu,
    ind2ptr_cpu,
    non_diag_mask_cpu,
    ptr2ind_cpu,
    radix_sort_cpu,
    random_walk_cpu,
    scatter_cpu,
    segment_csr_cpu,
    spmm_cpu,
    spmm_value_bw_cpu,
    spspmm_count_cpu,
    spspmm_fill_cpu,
    subgraph_count,
    subgraph_fill,
)


# =========================================================== convert_cpu.cpp
@export("mts_ind2ptr")
def mts_ind2ptr(ind: Int, dst: Int, numel: Int, num_rows: Int) abi("C"):
    ind2ptr_cpu(CPtr[DType.int64](unsafe_from_address=ind), IPtr(unsafe_from_address=dst), numel, num_rows)


@export("mts_ptr2ind")
def mts_ptr2ind(ptr: Int, dst: Int, numel: Int, numel_out: Int) abi("C"):
    ptr2ind_cpu(CPtr[DType.int64](unsafe_from_address=ptr), IPtr(unsafe_from_address=dst), numel, numel_out)


# ================================================= torch_sparse/utils.py
@export("mts_radix_sort")
def mts_radix_sort(
    keys: Int, payload: Int, keys_out: Int, payload_out: Int, counts: Int, n: Int,
) abi("C"):
    radix_sort_cpu(
        IPtr(unsafe_from_address=keys), IPtr(unsafe_from_address=payload),
        IPtr(unsafe_from_address=keys_out), IPtr(unsafe_from_address=payload_out),
        IPtr(unsafe_from_address=counts), n,
    )


# ============================================================= torch_scatter
@export("mts_gather_csr_f64")
def mts_gather_csr_f64(src: Int, indptr: Int, dst: Int, numel: Int) abi("C"):
    gather_csr_cpu[DType.float64](CPtr[DType.float64](unsafe_from_address=src), CPtr[DType.int64](unsafe_from_address=indptr), Ptr[DType.float64](unsafe_from_address=dst), numel)


@export("mts_gather_csr_f32")
def mts_gather_csr_f32(src: Int, indptr: Int, dst: Int, numel: Int) abi("C"):
    gather_csr_cpu[DType.float32](CPtr[DType.float32](unsafe_from_address=src), CPtr[DType.int64](unsafe_from_address=indptr), Ptr[DType.float32](unsafe_from_address=dst), numel)


@export("mts_segment_csr_f64_sum")
def mts_segment_csr_f64_sum(src: Int, indptr: Int, dst: Int, arg_out: Int, nseg: Int, scratch: Int, arg_scratch: Int) abi("C"):
    segment_csr_cpu[DType.float64, SUM](CPtr[DType.float64](unsafe_from_address=src), CPtr[DType.int64](unsafe_from_address=indptr), Ptr[DType.float64](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), nseg, Ptr[DType.float64](unsafe_from_address=scratch), IPtr(unsafe_from_address=arg_scratch))


@export("mts_segment_csr_f64_mean")
def mts_segment_csr_f64_mean(src: Int, indptr: Int, dst: Int, arg_out: Int, nseg: Int, scratch: Int, arg_scratch: Int) abi("C"):
    segment_csr_cpu[DType.float64, MEAN](CPtr[DType.float64](unsafe_from_address=src), CPtr[DType.int64](unsafe_from_address=indptr), Ptr[DType.float64](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), nseg, Ptr[DType.float64](unsafe_from_address=scratch), IPtr(unsafe_from_address=arg_scratch))


@export("mts_segment_csr_f64_min")
def mts_segment_csr_f64_min(src: Int, indptr: Int, dst: Int, arg_out: Int, nseg: Int, scratch: Int, arg_scratch: Int) abi("C"):
    segment_csr_cpu[DType.float64, MIN](CPtr[DType.float64](unsafe_from_address=src), CPtr[DType.int64](unsafe_from_address=indptr), Ptr[DType.float64](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), nseg, Ptr[DType.float64](unsafe_from_address=scratch), IPtr(unsafe_from_address=arg_scratch))


@export("mts_segment_csr_f64_max")
def mts_segment_csr_f64_max(src: Int, indptr: Int, dst: Int, arg_out: Int, nseg: Int, scratch: Int, arg_scratch: Int) abi("C"):
    segment_csr_cpu[DType.float64, MAX](CPtr[DType.float64](unsafe_from_address=src), CPtr[DType.int64](unsafe_from_address=indptr), Ptr[DType.float64](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), nseg, Ptr[DType.float64](unsafe_from_address=scratch), IPtr(unsafe_from_address=arg_scratch))


@export("mts_segment_csr_f32_sum")
def mts_segment_csr_f32_sum(src: Int, indptr: Int, dst: Int, arg_out: Int, nseg: Int, scratch: Int, arg_scratch: Int) abi("C"):
    segment_csr_cpu[DType.float32, SUM](CPtr[DType.float32](unsafe_from_address=src), CPtr[DType.int64](unsafe_from_address=indptr), Ptr[DType.float32](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), nseg, Ptr[DType.float32](unsafe_from_address=scratch), IPtr(unsafe_from_address=arg_scratch))


@export("mts_segment_csr_f32_mean")
def mts_segment_csr_f32_mean(src: Int, indptr: Int, dst: Int, arg_out: Int, nseg: Int, scratch: Int, arg_scratch: Int) abi("C"):
    segment_csr_cpu[DType.float32, MEAN](CPtr[DType.float32](unsafe_from_address=src), CPtr[DType.int64](unsafe_from_address=indptr), Ptr[DType.float32](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), nseg, Ptr[DType.float32](unsafe_from_address=scratch), IPtr(unsafe_from_address=arg_scratch))


@export("mts_segment_csr_f32_min")
def mts_segment_csr_f32_min(src: Int, indptr: Int, dst: Int, arg_out: Int, nseg: Int, scratch: Int, arg_scratch: Int) abi("C"):
    segment_csr_cpu[DType.float32, MIN](CPtr[DType.float32](unsafe_from_address=src), CPtr[DType.int64](unsafe_from_address=indptr), Ptr[DType.float32](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), nseg, Ptr[DType.float32](unsafe_from_address=scratch), IPtr(unsafe_from_address=arg_scratch))


@export("mts_segment_csr_f32_max")
def mts_segment_csr_f32_max(src: Int, indptr: Int, dst: Int, arg_out: Int, nseg: Int, scratch: Int, arg_scratch: Int) abi("C"):
    segment_csr_cpu[DType.float32, MAX](CPtr[DType.float32](unsafe_from_address=src), CPtr[DType.int64](unsafe_from_address=indptr), Ptr[DType.float32](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), nseg, Ptr[DType.float32](unsafe_from_address=scratch), IPtr(unsafe_from_address=arg_scratch))


@export("mts_scatter_f64_sum")
def mts_scatter_f64_sum(src: Int, index: Int, dst: Int, arg_out: Int, counts: Int, dim_size: Int, n: Int, stride: Int) abi("C"):
    scatter_cpu[DType.float64, SUM](CPtr[DType.float64](unsafe_from_address=src), CPtr[DType.int64](unsafe_from_address=index), Ptr[DType.float64](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), IPtr(unsafe_from_address=counts), dim_size, n, stride)


@export("mts_scatter_f64_mean")
def mts_scatter_f64_mean(src: Int, index: Int, dst: Int, arg_out: Int, counts: Int, dim_size: Int, n: Int, stride: Int) abi("C"):
    scatter_cpu[DType.float64, MEAN](CPtr[DType.float64](unsafe_from_address=src), CPtr[DType.int64](unsafe_from_address=index), Ptr[DType.float64](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), IPtr(unsafe_from_address=counts), dim_size, n, stride)


@export("mts_scatter_f64_min")
def mts_scatter_f64_min(src: Int, index: Int, dst: Int, arg_out: Int, counts: Int, dim_size: Int, n: Int, stride: Int) abi("C"):
    scatter_cpu[DType.float64, MIN](CPtr[DType.float64](unsafe_from_address=src), CPtr[DType.int64](unsafe_from_address=index), Ptr[DType.float64](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), IPtr(unsafe_from_address=counts), dim_size, n, stride)


@export("mts_scatter_f64_max")
def mts_scatter_f64_max(src: Int, index: Int, dst: Int, arg_out: Int, counts: Int, dim_size: Int, n: Int, stride: Int) abi("C"):
    scatter_cpu[DType.float64, MAX](CPtr[DType.float64](unsafe_from_address=src), CPtr[DType.int64](unsafe_from_address=index), Ptr[DType.float64](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), IPtr(unsafe_from_address=counts), dim_size, n, stride)


@export("mts_scatter_f32_sum")
def mts_scatter_f32_sum(src: Int, index: Int, dst: Int, arg_out: Int, counts: Int, dim_size: Int, n: Int, stride: Int) abi("C"):
    scatter_cpu[DType.float32, SUM](CPtr[DType.float32](unsafe_from_address=src), CPtr[DType.int64](unsafe_from_address=index), Ptr[DType.float32](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), IPtr(unsafe_from_address=counts), dim_size, n, stride)


@export("mts_scatter_f32_mean")
def mts_scatter_f32_mean(src: Int, index: Int, dst: Int, arg_out: Int, counts: Int, dim_size: Int, n: Int, stride: Int) abi("C"):
    scatter_cpu[DType.float32, MEAN](CPtr[DType.float32](unsafe_from_address=src), CPtr[DType.int64](unsafe_from_address=index), Ptr[DType.float32](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), IPtr(unsafe_from_address=counts), dim_size, n, stride)


@export("mts_scatter_f32_min")
def mts_scatter_f32_min(src: Int, index: Int, dst: Int, arg_out: Int, counts: Int, dim_size: Int, n: Int, stride: Int) abi("C"):
    scatter_cpu[DType.float32, MIN](CPtr[DType.float32](unsafe_from_address=src), CPtr[DType.int64](unsafe_from_address=index), Ptr[DType.float32](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), IPtr(unsafe_from_address=counts), dim_size, n, stride)


@export("mts_scatter_f32_max")
def mts_scatter_f32_max(src: Int, index: Int, dst: Int, arg_out: Int, counts: Int, dim_size: Int, n: Int, stride: Int) abi("C"):
    scatter_cpu[DType.float32, MAX](CPtr[DType.float32](unsafe_from_address=src), CPtr[DType.int64](unsafe_from_address=index), Ptr[DType.float32](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), IPtr(unsafe_from_address=counts), dim_size, n, stride)


# =========================================================== spmm_cpu.cpp
@export("mts_spmm_f64_sum")
def mts_spmm_f64_sum(rowptr: Int, col: Int, value: Int, mat: Int, dst: Int, arg_out: Int, vals: Int, args: Int, num_rows: Int, num_inner: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_cpu[DType.float64, SUM, True](CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float64](unsafe_from_address=value), CPtr[DType.float64](unsafe_from_address=mat), Ptr[DType.float64](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), Ptr[DType.float64](unsafe_from_address=vals), IPtr(unsafe_from_address=args), num_rows, num_inner, num_cols, batch)


@export("mts_spmm_f64_mean")
def mts_spmm_f64_mean(rowptr: Int, col: Int, value: Int, mat: Int, dst: Int, arg_out: Int, vals: Int, args: Int, num_rows: Int, num_inner: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_cpu[DType.float64, MEAN, True](CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float64](unsafe_from_address=value), CPtr[DType.float64](unsafe_from_address=mat), Ptr[DType.float64](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), Ptr[DType.float64](unsafe_from_address=vals), IPtr(unsafe_from_address=args), num_rows, num_inner, num_cols, batch)


@export("mts_spmm_f64_min")
def mts_spmm_f64_min(rowptr: Int, col: Int, value: Int, mat: Int, dst: Int, arg_out: Int, vals: Int, args: Int, num_rows: Int, num_inner: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_cpu[DType.float64, MIN, True](CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float64](unsafe_from_address=value), CPtr[DType.float64](unsafe_from_address=mat), Ptr[DType.float64](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), Ptr[DType.float64](unsafe_from_address=vals), IPtr(unsafe_from_address=args), num_rows, num_inner, num_cols, batch)


@export("mts_spmm_f64_max")
def mts_spmm_f64_max(rowptr: Int, col: Int, value: Int, mat: Int, dst: Int, arg_out: Int, vals: Int, args: Int, num_rows: Int, num_inner: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_cpu[DType.float64, MAX, True](CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float64](unsafe_from_address=value), CPtr[DType.float64](unsafe_from_address=mat), Ptr[DType.float64](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), Ptr[DType.float64](unsafe_from_address=vals), IPtr(unsafe_from_address=args), num_rows, num_inner, num_cols, batch)


@export("mts_spmm_f32_sum")
def mts_spmm_f32_sum(rowptr: Int, col: Int, value: Int, mat: Int, dst: Int, arg_out: Int, vals: Int, args: Int, num_rows: Int, num_inner: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_cpu[DType.float32, SUM, True](CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float32](unsafe_from_address=value), CPtr[DType.float32](unsafe_from_address=mat), Ptr[DType.float32](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), Ptr[DType.float32](unsafe_from_address=vals), IPtr(unsafe_from_address=args), num_rows, num_inner, num_cols, batch)


@export("mts_spmm_f32_mean")
def mts_spmm_f32_mean(rowptr: Int, col: Int, value: Int, mat: Int, dst: Int, arg_out: Int, vals: Int, args: Int, num_rows: Int, num_inner: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_cpu[DType.float32, MEAN, True](CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float32](unsafe_from_address=value), CPtr[DType.float32](unsafe_from_address=mat), Ptr[DType.float32](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), Ptr[DType.float32](unsafe_from_address=vals), IPtr(unsafe_from_address=args), num_rows, num_inner, num_cols, batch)


@export("mts_spmm_f32_min")
def mts_spmm_f32_min(rowptr: Int, col: Int, value: Int, mat: Int, dst: Int, arg_out: Int, vals: Int, args: Int, num_rows: Int, num_inner: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_cpu[DType.float32, MIN, True](CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float32](unsafe_from_address=value), CPtr[DType.float32](unsafe_from_address=mat), Ptr[DType.float32](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), Ptr[DType.float32](unsafe_from_address=vals), IPtr(unsafe_from_address=args), num_rows, num_inner, num_cols, batch)


@export("mts_spmm_f32_max")
def mts_spmm_f32_max(rowptr: Int, col: Int, value: Int, mat: Int, dst: Int, arg_out: Int, vals: Int, args: Int, num_rows: Int, num_inner: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_cpu[DType.float32, MAX, True](CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float32](unsafe_from_address=value), CPtr[DType.float32](unsafe_from_address=mat), Ptr[DType.float32](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), Ptr[DType.float32](unsafe_from_address=vals), IPtr(unsafe_from_address=args), num_rows, num_inner, num_cols, batch)


@export("mts_spmm_f64_nv_sum")
def mts_spmm_f64_nv_sum(rowptr: Int, col: Int, value: Int, mat: Int, dst: Int, arg_out: Int, vals: Int, args: Int, num_rows: Int, num_inner: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_cpu[DType.float64, SUM, False](CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float64](unsafe_from_address=value), CPtr[DType.float64](unsafe_from_address=mat), Ptr[DType.float64](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), Ptr[DType.float64](unsafe_from_address=vals), IPtr(unsafe_from_address=args), num_rows, num_inner, num_cols, batch)


@export("mts_spmm_f64_nv_mean")
def mts_spmm_f64_nv_mean(rowptr: Int, col: Int, value: Int, mat: Int, dst: Int, arg_out: Int, vals: Int, args: Int, num_rows: Int, num_inner: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_cpu[DType.float64, MEAN, False](CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float64](unsafe_from_address=value), CPtr[DType.float64](unsafe_from_address=mat), Ptr[DType.float64](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), Ptr[DType.float64](unsafe_from_address=vals), IPtr(unsafe_from_address=args), num_rows, num_inner, num_cols, batch)


@export("mts_spmm_f64_nv_min")
def mts_spmm_f64_nv_min(rowptr: Int, col: Int, value: Int, mat: Int, dst: Int, arg_out: Int, vals: Int, args: Int, num_rows: Int, num_inner: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_cpu[DType.float64, MIN, False](CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float64](unsafe_from_address=value), CPtr[DType.float64](unsafe_from_address=mat), Ptr[DType.float64](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), Ptr[DType.float64](unsafe_from_address=vals), IPtr(unsafe_from_address=args), num_rows, num_inner, num_cols, batch)


@export("mts_spmm_f64_nv_max")
def mts_spmm_f64_nv_max(rowptr: Int, col: Int, value: Int, mat: Int, dst: Int, arg_out: Int, vals: Int, args: Int, num_rows: Int, num_inner: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_cpu[DType.float64, MAX, False](CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float64](unsafe_from_address=value), CPtr[DType.float64](unsafe_from_address=mat), Ptr[DType.float64](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), Ptr[DType.float64](unsafe_from_address=vals), IPtr(unsafe_from_address=args), num_rows, num_inner, num_cols, batch)


@export("mts_spmm_f32_nv_sum")
def mts_spmm_f32_nv_sum(rowptr: Int, col: Int, value: Int, mat: Int, dst: Int, arg_out: Int, vals: Int, args: Int, num_rows: Int, num_inner: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_cpu[DType.float32, SUM, False](CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float32](unsafe_from_address=value), CPtr[DType.float32](unsafe_from_address=mat), Ptr[DType.float32](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), Ptr[DType.float32](unsafe_from_address=vals), IPtr(unsafe_from_address=args), num_rows, num_inner, num_cols, batch)


@export("mts_spmm_f32_nv_mean")
def mts_spmm_f32_nv_mean(rowptr: Int, col: Int, value: Int, mat: Int, dst: Int, arg_out: Int, vals: Int, args: Int, num_rows: Int, num_inner: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_cpu[DType.float32, MEAN, False](CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float32](unsafe_from_address=value), CPtr[DType.float32](unsafe_from_address=mat), Ptr[DType.float32](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), Ptr[DType.float32](unsafe_from_address=vals), IPtr(unsafe_from_address=args), num_rows, num_inner, num_cols, batch)


@export("mts_spmm_f32_nv_min")
def mts_spmm_f32_nv_min(rowptr: Int, col: Int, value: Int, mat: Int, dst: Int, arg_out: Int, vals: Int, args: Int, num_rows: Int, num_inner: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_cpu[DType.float32, MIN, False](CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float32](unsafe_from_address=value), CPtr[DType.float32](unsafe_from_address=mat), Ptr[DType.float32](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), Ptr[DType.float32](unsafe_from_address=vals), IPtr(unsafe_from_address=args), num_rows, num_inner, num_cols, batch)


@export("mts_spmm_f32_nv_max")
def mts_spmm_f32_nv_max(rowptr: Int, col: Int, value: Int, mat: Int, dst: Int, arg_out: Int, vals: Int, args: Int, num_rows: Int, num_inner: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_cpu[DType.float32, MAX, False](CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float32](unsafe_from_address=value), CPtr[DType.float32](unsafe_from_address=mat), Ptr[DType.float32](unsafe_from_address=dst), IPtr(unsafe_from_address=arg_out), Ptr[DType.float32](unsafe_from_address=vals), IPtr(unsafe_from_address=args), num_rows, num_inner, num_cols, batch)


@export("mts_spmm_value_bw_f64_sum")
def mts_spmm_value_bw_f64_sum(row: Int, rowptr: Int, col: Int, mat: Int, grad: Int, dst: Int, num_rows: Int, num_inner: Int, numel: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_value_bw_cpu[DType.float64, SUM](CPtr[DType.int64](unsafe_from_address=row), CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float64](unsafe_from_address=mat), CPtr[DType.float64](unsafe_from_address=grad), Ptr[DType.float64](unsafe_from_address=dst), num_rows, num_inner, numel, num_cols, batch)


@export("mts_spmm_value_bw_f64_mean")
def mts_spmm_value_bw_f64_mean(row: Int, rowptr: Int, col: Int, mat: Int, grad: Int, dst: Int, num_rows: Int, num_inner: Int, numel: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_value_bw_cpu[DType.float64, MEAN](CPtr[DType.int64](unsafe_from_address=row), CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float64](unsafe_from_address=mat), CPtr[DType.float64](unsafe_from_address=grad), Ptr[DType.float64](unsafe_from_address=dst), num_rows, num_inner, numel, num_cols, batch)


@export("mts_spmm_value_bw_f32_sum")
def mts_spmm_value_bw_f32_sum(row: Int, rowptr: Int, col: Int, mat: Int, grad: Int, dst: Int, num_rows: Int, num_inner: Int, numel: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_value_bw_cpu[DType.float32, SUM](CPtr[DType.int64](unsafe_from_address=row), CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float32](unsafe_from_address=mat), CPtr[DType.float32](unsafe_from_address=grad), Ptr[DType.float32](unsafe_from_address=dst), num_rows, num_inner, numel, num_cols, batch)


@export("mts_spmm_value_bw_f32_mean")
def mts_spmm_value_bw_f32_mean(row: Int, rowptr: Int, col: Int, mat: Int, grad: Int, dst: Int, num_rows: Int, num_inner: Int, numel: Int, num_cols: Int, batch: Int) abi("C"):
    spmm_value_bw_cpu[DType.float32, MEAN](CPtr[DType.int64](unsafe_from_address=row), CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.float32](unsafe_from_address=mat), CPtr[DType.float32](unsafe_from_address=grad), Ptr[DType.float32](unsafe_from_address=dst), num_rows, num_inner, numel, num_cols, batch)


# ============================================================== rw_cpu.cpp
@export("mts_random_walk")
def mts_random_walk(rowptr: Int, col: Int, start: Int, rand: Int, dst: Int, batch: Int, walk_length: Int) abi("C"):
    random_walk_cpu(CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.int64](unsafe_from_address=start), F32CPtr(unsafe_from_address=rand), IPtr(unsafe_from_address=dst), batch, walk_length)


# ============================================================ diag_cpu.cpp
@export("mts_non_diag_mask")
def mts_non_diag_mask(row: Int, col: Int, mask: Int, numel: Int, num_rows: Int, num_cols: Int, k: Int) abi("C"):
    non_diag_mask_cpu(CPtr[DType.int64](unsafe_from_address=row), CPtr[DType.int64](unsafe_from_address=col), BPtr(unsafe_from_address=mask), numel, num_rows, num_cols, k)


# =========================================================== saint_cpu.cpp
@export("mts_saint_subgraph_count")
def mts_saint_subgraph_count(idx: Int, rowptr: Int, col: Int, assoc: Int, length_out: Int, num_idx: Int) abi("C"):
    subgraph_count(CPtr[DType.int64](unsafe_from_address=idx), CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.int64](unsafe_from_address=assoc), IPtr(unsafe_from_address=length_out), num_idx)


@export("mts_saint_subgraph_fill")
def mts_saint_subgraph_fill(idx: Int, rowptr: Int, col: Int, assoc: Int, row_out: Int, col_out: Int, idx_out: Int, num_idx: Int) abi("C"):
    subgraph_fill(CPtr[DType.int64](unsafe_from_address=idx), CPtr[DType.int64](unsafe_from_address=rowptr), CPtr[DType.int64](unsafe_from_address=col), CPtr[DType.int64](unsafe_from_address=assoc), IPtr(unsafe_from_address=row_out), IPtr(unsafe_from_address=col_out), IPtr(unsafe_from_address=idx_out), num_idx)


# ====================================================== spspmm (matmul)
@export("mts_spspmm_count")
def mts_spspmm_count(rowptr_a: Int, col_a: Int, rowptr_b: Int, count_out: Int, num_rows: Int) abi("C"):
    spspmm_count_cpu(CPtr[DType.int64](unsafe_from_address=rowptr_a), CPtr[DType.int64](unsafe_from_address=col_a), CPtr[DType.int64](unsafe_from_address=rowptr_b), IPtr(unsafe_from_address=count_out), num_rows)


@export("mts_spspmm_fill_f64")
def mts_spspmm_fill_f64(rowptr_a: Int, col_a: Int, value_a: Int, rowptr_b: Int, col_b: Int, value_b: Int, out_rowptr: Int, out_col: Int, out_value: Int, num_rows: Int) abi("C"):
    spspmm_fill_cpu[DType.float64, True](CPtr[DType.int64](unsafe_from_address=rowptr_a), CPtr[DType.int64](unsafe_from_address=col_a), CPtr[DType.float64](unsafe_from_address=value_a), CPtr[DType.int64](unsafe_from_address=rowptr_b), CPtr[DType.int64](unsafe_from_address=col_b), CPtr[DType.float64](unsafe_from_address=value_b), IPtr(unsafe_from_address=out_rowptr), IPtr(unsafe_from_address=out_col), Ptr[DType.float64](unsafe_from_address=out_value), num_rows)


@export("mts_spspmm_fill_f32")
def mts_spspmm_fill_f32(rowptr_a: Int, col_a: Int, value_a: Int, rowptr_b: Int, col_b: Int, value_b: Int, out_rowptr: Int, out_col: Int, out_value: Int, num_rows: Int) abi("C"):
    spspmm_fill_cpu[DType.float32, True](CPtr[DType.int64](unsafe_from_address=rowptr_a), CPtr[DType.int64](unsafe_from_address=col_a), CPtr[DType.float32](unsafe_from_address=value_a), CPtr[DType.int64](unsafe_from_address=rowptr_b), CPtr[DType.int64](unsafe_from_address=col_b), CPtr[DType.float32](unsafe_from_address=value_b), IPtr(unsafe_from_address=out_rowptr), IPtr(unsafe_from_address=out_col), Ptr[DType.float32](unsafe_from_address=out_value), num_rows)


@export("mts_spspmm_fill_f64_nv")
def mts_spspmm_fill_f64_nv(rowptr_a: Int, col_a: Int, value_a: Int, rowptr_b: Int, col_b: Int, value_b: Int, out_rowptr: Int, out_col: Int, out_value: Int, num_rows: Int) abi("C"):
    spspmm_fill_cpu[DType.float64, False](CPtr[DType.int64](unsafe_from_address=rowptr_a), CPtr[DType.int64](unsafe_from_address=col_a), CPtr[DType.float64](unsafe_from_address=value_a), CPtr[DType.int64](unsafe_from_address=rowptr_b), CPtr[DType.int64](unsafe_from_address=col_b), CPtr[DType.float64](unsafe_from_address=value_b), IPtr(unsafe_from_address=out_rowptr), IPtr(unsafe_from_address=out_col), Ptr[DType.float64](unsafe_from_address=out_value), num_rows)


@export("mts_spspmm_fill_f32_nv")
def mts_spspmm_fill_f32_nv(rowptr_a: Int, col_a: Int, value_a: Int, rowptr_b: Int, col_b: Int, value_b: Int, out_rowptr: Int, out_col: Int, out_value: Int, num_rows: Int) abi("C"):
    spspmm_fill_cpu[DType.float32, False](CPtr[DType.int64](unsafe_from_address=rowptr_a), CPtr[DType.int64](unsafe_from_address=col_a), CPtr[DType.float32](unsafe_from_address=value_a), CPtr[DType.int64](unsafe_from_address=rowptr_b), CPtr[DType.int64](unsafe_from_address=col_b), CPtr[DType.float32](unsafe_from_address=value_b), IPtr(unsafe_from_address=out_rowptr), IPtr(unsafe_from_address=out_col), Ptr[DType.float32](unsafe_from_address=out_value), num_rows)
