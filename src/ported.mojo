"""torch-sparse kernels, ported to Mojo in upstream source order.

Read this file top to bottom against upstream: every section is named after the
file it was ported from, and the functions keep upstream's names, argument
order and control flow. Section order follows `torch_sparse/__init__.py`, so it
also matches the order the C++ kernels are reached from:

    csrc/cpu/convert_cpu.cpp   ind2ptr_cpu, ptr2ind_cpu
    csrc/cpu/reducer.h        Reducer<scalar_t, REDUCE>
    torch_sparse/utils.py      index_sort
    torch_scatter             gather_csr, segment_csr, scatter
    csrc/cpu/spmm_cpu.cpp     spmm_cpu, spmm_value_bw_cpu
    csrc/cpu/rw_cpu.cpp       random_walk_cpu
    csrc/cpu/diag_cpu.cpp     non_diag_mask_cpu
    csrc/cpu/saint_cpu.cpp    subgraph_cpu
    torch.sparse.mm           the SpGEMM behind SparseTensor.spspmm

Two dialect decisions are load-bearing and are documented in MOJO_NOTES.md:

* `@export` refuses parametric functions, so the kernels here *are* parametric
  (over the element type and the reduction) and `src/capi.mojo` instantiates
  them concretely. The upstream structure therefore survives in the kernel
  rather than being flattened into the ABI layer.
* Nothing allocates. Upstream's `std::vector<scalar_t> vals(num_cols)` scratch becomes
  a caller-owned buffer whose address the ctypes glue passes in; the loop that
  fills and drains it is unchanged.

`Int` is the index/address type; `Int64` is the payload type of the int64 index
buffers, and `Int(x)` narrows it back. Mixing the two silently miscompiles
(`SIMD[dtype, w] * Int` does not resolve), so index reads are cast at the point
of the load.
"""

# `csrc/cpu/reducer.h`: `enum ReductionType { SUM, MEAN, MUL, DIV, MIN, MAX }`.
comptime SUM: Int = 0
comptime MEAN: Int = 1
comptime MUL: Int = 2
comptime DIV: Int = 3
comptime MIN: Int = 4
comptime MAX: Int = 5

# Stand-ins for `std::numeric_limits<scalar_t>`; the kernels are instantiated
# for float32 and float64 only, and the wider bound is harmless in both.
comptime LARGEST: Float64 = 1.7976931348623157e308
comptime SIGN_BIT: Int64 = 0x8000000000000000

comptime Ptr[dtype: DType] = Pointer[Scalar[dtype], AnyOrigin[mut=True]]
comptime CPtr[dtype: DType] = Pointer[Scalar[dtype], AnyOrigin[mut=False]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime CIPtr = Pointer[Int64, AnyOrigin[mut=False]]
comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime F32CPtr = Pointer[Float32, AnyOrigin[mut=False]]


# =========================================================== convert_cpu.cpp
def ind2ptr_cpu(ind: CIPtr, dst: IPtr, numel: Int, num_rows: Int):
    """`ind2ptr_cpu(ind, M)`: row ids -> row pointer of length `M + 1`."""
    if numel == 0:
        for i in range(num_rows + 1):
            dst.unsafe_store(i, Int64(0))
        return

    for i in range(Int(ind.unsafe_load(0)) + 1):
        dst.unsafe_store(i, Int64(0))

    # upstream splits this over threads; the range visited is identical.
    for i in range(numel - 1):
        var idx = Int(ind.unsafe_load(i))
        var next_idx = Int(ind.unsafe_load(i + 1))
        while idx < next_idx:
            dst.unsafe_store(idx + 1, Int64(i + 1))
            idx += 1

    for i in range(Int(ind.unsafe_load(numel - 1)) + 1, num_rows + 1):
        dst.unsafe_store(i, Int64(numel))


def ptr2ind_cpu(ptr: CIPtr, dst: IPtr, numel: Int, nnz: Int):
    """`ptr2ind_cpu(ptr, E)`: row pointer -> row id of length `E`."""
    for i in range(numel - 1):
        var idx = Int(ptr.unsafe_load(i))
        var next_idx = Int(ptr.unsafe_load(i + 1))
        while idx < next_idx:
            dst.unsafe_store(idx, Int64(i))
            idx += 1


# ================================================================ reducer.h
def reducer_init[dtype: DType, red: Int]() -> Scalar[dtype]:
    if red == MUL or red == DIV:
        return Scalar[dtype](1)
    elif red == MIN:
        return Scalar[dtype](LARGEST)
    elif red == MAX:
        return Scalar[dtype](-LARGEST)
    else:
        return Scalar[dtype](0)


def reducer_update[dtype: DType, red: Int](
    val: Ptr[dtype], new_val: Scalar[dtype], arg: IPtr, new_arg: Int
):
    """`Reducer::update`; upstream passes the accumulator and the arg slot by
    address, and so does this."""
    if red == SUM or red == MEAN:
        val.unsafe_store(0, val.unsafe_load(0) + new_val)
    elif red == MUL:
        val.unsafe_store(0, val.unsafe_load(0) * new_val)
    elif red == DIV:
        val.unsafe_store(0, val.unsafe_load(0) / new_val)
    elif (red == MIN and new_val < val.unsafe_load(0)) or (
        red == MAX and new_val > val.unsafe_load(0)
    ):
        val.unsafe_store(0, new_val)
        arg.unsafe_store(0, Int64(new_arg))


def reducer_write[dtype: DType, red: Int](
    address: Ptr[dtype], val: Scalar[dtype], arg_address: IPtr, arg: Int, count: Int
):
    """`Reducer::write`."""
    if red == SUM or red == MUL or red == DIV:
        address.unsafe_store(0, val)
    elif red == MEAN:
        address.unsafe_store(0, val / Scalar[dtype](count if count > 0 else 1))
    elif red == MIN or red == MAX:
        if count > 0:
            address.unsafe_store(0, val)
            arg_address.unsafe_store(0, Int64(arg))
        else:
            address.unsafe_store(0, Scalar[dtype](0))


# ================================================= torch_sparse/utils.py
def radix_sort_cpu(
    keys: IPtr, payload: IPtr, keys_out: IPtr, payload_out: IPtr,
    counts: IPtr, n: Int,
):
    """A stable LSD radix sort on signed 64-bit keys, carrying `payload`.

    `torch_sparse.utils.index_sort` is pyg-lib's counting sort. A *stable*
    sort by the same key produces the identical permutation, and a radix sort
    keeps working when the key range is `M * N` rather than `N`. `counts` is
    a caller-owned 256-entry histogram.
    """
    # Bucketing flips the sign bit so that the unsigned byte order of the
    # result is the signed order of the key; the stored key stays the original.
    var src_keys = keys
    var src_payload = payload
    var dst_keys = keys_out
    var dst_payload = payload_out
    # Eight passes alternate between the two buffer pairs an even number of
    # times, which leaves the sorted run back in the *input* pair; track that
    # and copy it out, rather than assuming where it landed.
    var back_in_src = True

    var shift: Int = 0
    while shift < 64:
        for b in range(256):
            counts.unsafe_store(b, Int64(0))
        for i in range(n):
            var b = Int(
                ((src_keys.unsafe_load(i) ^ SIGN_BIT) >> Int64(shift)) & Int64(0xFF)
            )
            counts.unsafe_store(b, counts.unsafe_load(b) + Int64(1))
        var total = Int64(0)
        for b in range(256):
            var c = counts.unsafe_load(b)
            counts.unsafe_store(b, total)
            total += c
        for i in range(n):
            var b = Int(
                ((src_keys.unsafe_load(i) ^ SIGN_BIT) >> Int64(shift)) & Int64(0xFF)
            )
            var pos = Int(counts.unsafe_load(b))
            counts.unsafe_store(b, Int64(pos + 1))
            dst_keys.unsafe_store(pos, src_keys.unsafe_load(i))
            dst_payload.unsafe_store(pos, src_payload.unsafe_load(i))

        var t = src_keys
        src_keys = dst_keys
        dst_keys = t
        t = src_payload
        src_payload = dst_payload
        dst_payload = t
        back_in_src = not back_in_src
        shift += 8

    if back_in_src:
        for i in range(n):
            keys_out.unsafe_store(i, src_keys.unsafe_load(i))
            payload_out.unsafe_store(i, src_payload.unsafe_load(i))


# ============================================================= torch_scatter
def gather_csr_cpu[dtype: DType](
    src: CPtr[dtype], indptr: CIPtr, dst: Ptr[dtype], numel: Int
):
    """`torch_scatter.gather_csr(src, indptr)`: expand by segment length."""
    for i in range(numel - 1):
        var start = Int(indptr.unsafe_load(i))
        var end = Int(indptr.unsafe_load(i + 1))
        for e in range(start, end):
            dst.unsafe_store(e, src.unsafe_load(i))


def segment_csr_cpu[dtype: DType, red: Int](
    src: CPtr[dtype], indptr: CIPtr, dst: Ptr[dtype], arg_out: IPtr,
    nseg: Int, scratch: Ptr[dtype], arg_scratch: IPtr,
):
    """`torch_scatter.segment_csr(src, indptr, dst, reduce)`.

    `scratch`/`arg_scratch` are the caller-owned accumulators that upstream
    holds in a register and a vector; only `min`/`max` touch the arg slot.
    """
    for i in range(nseg):
        var start = Int(indptr.unsafe_load(i))
        var end = Int(indptr.unsafe_load(i + 1))

        scratch.unsafe_store(0, reducer_init[dtype, red]())
        arg_scratch.unsafe_store(0, Int64(-1))

        for e in range(start, end):
            reducer_update[dtype, red](scratch, src.unsafe_load(e), arg_scratch, e)

        reducer_write[dtype, red](
            dst.unsafe_offset(i), scratch.unsafe_load(0), arg_out.unsafe_offset(i),
            Int(arg_scratch.unsafe_load(0)), end - start,
        )


def scatter_cpu[dtype: DType, red: Int](
    src: CPtr[dtype], index: CIPtr, dst: Ptr[dtype], arg_out: IPtr,
    counts: IPtr, dim_size: Int, n: Int, stride: Int,
):
    """`torch_scatter.scatter(src, index, dim, out, dim_size, reduce)`.

    `src` is `n x stride` and the destination `dim_size x stride`, which is
    upstream's `src.stride(0)` with the trailing dimensions carried whole;
    `counts` is a caller-owned int64 buffer of length `dim_size`.
    """
    for i in range(dim_size * stride):
        dst.unsafe_store(i, reducer_init[dtype, red]())
        arg_out.unsafe_store(i, Int64(-1))
    for i in range(dim_size):
        counts.unsafe_store(i, Int64(0))

    for i in range(n):
        var j = Int(index.unsafe_load(i))
        for s in range(stride):
            reducer_update[dtype, red](
                dst.unsafe_offset(j * stride + s), src.unsafe_load(i * stride + s),
                arg_out.unsafe_offset(j * stride + s), i,
            )
        counts.unsafe_store(j, counts.unsafe_load(j) + Int64(1))

    for j in range(dim_size):
        for s in range(stride):
            reducer_write[dtype, red](
                dst.unsafe_offset(j * stride + s), dst.unsafe_load(j * stride + s),
                arg_out.unsafe_offset(j * stride + s),
                Int(arg_out.unsafe_load(j * stride + s)),
                Int(counts.unsafe_load(j)),
            )


# =========================================================== spmm_cpu.cpp
def spmm_cpu[dtype: DType, red: Int, has_value: Bool](
    rowptr: CIPtr, col: CIPtr, value: CPtr[dtype], mat: CPtr[dtype],
    dst: Ptr[dtype], arg_out: IPtr, vals: Ptr[dtype], args: IPtr,
    num_rows: Int, num_inner: Int, num_cols: Int, batch: Int,
):
    """`spmm_cpu(rowptr, col, optional_value, mat, reduce)`.

    Upstream takes `M = rowptr.numel() - 1`, `N = mat.size(-2)`,
    `K = mat.size(-1)` and `B = mat.numel() / (N * K)`, then loops over
    `B * M` inside an `at::parallel_for`. The parallel loop is a single serial
    loop here: this toolchain has no `parallelize`.
    """
    for i in range(batch * num_rows):
        var b = i / num_rows
        var m = i % num_rows

        var row_start = Int(rowptr.unsafe_load(m))
        var row_end = Int(rowptr.unsafe_load(m + 1))

        for k in range(num_cols):
            vals.unsafe_store(k, reducer_init[dtype, red]())
            args.unsafe_store(k, Int64(-1))

        var offset = b * num_inner * num_cols
        for e in range(row_start, row_end):
            var c = Int(col.unsafe_load(e))
            if has_value:
                var v = value.unsafe_load(e)
                for k in range(num_cols):
                    reducer_update[dtype, red](
                        vals.unsafe_offset(k), v * mat.unsafe_load(offset + c * num_cols + k),
                        args.unsafe_offset(k), e,
                    )
            else:
                for k in range(num_cols):
                    reducer_update[dtype, red](
                        vals.unsafe_offset(k), mat.unsafe_load(offset + c * num_cols + k),
                        args.unsafe_offset(k), e,
                    )

        offset = b * num_rows * num_cols + m * num_cols
        for k in range(num_cols):
            reducer_write[dtype, red](
                dst.unsafe_offset(offset + k), vals.unsafe_load(k),
                arg_out.unsafe_offset(offset + k), Int(args.unsafe_load(k)),
                row_end - row_start,
            )


def spmm_value_bw_cpu[dtype: DType, red: Int](
    row: CIPtr, rowptr: CIPtr, col: CIPtr, mat: CPtr[dtype], grad: CPtr[dtype],
    dst: Ptr[dtype], num_rows: Int, num_inner: Int, nnz: Int, num_cols: Int, batch: Int,
):
    """`spmm_value_bw_cpu(row, rowptr, col, mat, grad, reduce)`."""
    for e in range(nnz):
        dst.unsafe_store(e, Scalar[dtype](0))

    for b in range(batch):
        for e in range(nnz):
            var r = Int(row.unsafe_load(e))
            var c = Int(col.unsafe_load(e))
            var v = Scalar[dtype](0)
            for k in range(num_cols):
                v += mat.unsafe_load(b * num_inner * num_cols + c * num_cols + k) * grad.unsafe_load(
                    b * num_rows * num_cols + r * num_cols + k
                )
            if red == MEAN:
                var row_start = Int(rowptr.unsafe_load(r))
                var row_end = Int(rowptr.unsafe_load(r + 1))
                v /= Scalar[dtype](max(row_end - row_start, 1))
            dst.unsafe_store(e, dst.unsafe_load(e) + v)


# ============================================================== rw_cpu.cpp
def random_walk_cpu(
    rowptr: CIPtr, col: CIPtr, start: CIPtr, rand: F32CPtr,
    dst: IPtr, batch: Int, walk_length: Int,
):
    """`random_walk_cpu(rowptr, col, start, walk_length)`.

    `rand` is the `{start.size(0), walk_length}` uniform sample upstream draws
    from `torch::rand`. The Python side draws it instead, so the port is
    reproducible and the indexing below is upstream's unchanged.
    """
    var L = walk_length + 1
    for n in range(batch):
        var cur = Int(start.unsafe_load(n))
        dst.unsafe_store(n * L, Int64(cur))

        for l in range(walk_length):
            var row_start = Int(rowptr.unsafe_load(cur))
            var row_end = Int(rowptr.unsafe_load(cur + 1))

            # `int64_t(rand_data[...] * (row_end - row_start))`: the cast has to
            # follow the product. Truncating the sample first would floor it.
            var pick = Int(
                rand.unsafe_load(n * walk_length + l) * Float32(row_end - row_start)
            )
            cur = Int(col.unsafe_load(row_start + pick))
            dst.unsafe_store(n * L + l + 1, Int64(cur))


# ============================================================ diag_cpu.cpp
def non_diag_mask_cpu(row: CIPtr, col: CIPtr, mask: BPtr, numel: Int, num_rows: Int, num_inner: Int, k: Int):
    """`non_diag_mask_cpu(row, col, M, N, k)`.

    `mask` is the `E + num_diag` byte buffer that upstream allocates zeroed.
    """
    var num_diag = min(num_rows + k, num_inner) if k < 0 else min(num_rows, num_inner - k)

    if k < 0:
        for i in range(numel):
            var r = Int(row.unsafe_load(i))
            var c = Int(col.unsafe_load(i))
            if r + k < 0:
                mask.unsafe_store(i, UInt8(1))
            elif r + k >= num_inner:
                mask.unsafe_store(i + num_diag, UInt8(1))
            elif r + k > c:
                mask.unsafe_store(i + r + k, UInt8(1))
            elif r + k < c:
                mask.unsafe_store(i + r + k + 1, UInt8(1))
    else:
        for i in range(numel):
            var r = Int(row.unsafe_load(i))
            var c = Int(col.unsafe_load(i))
            if r + k >= num_inner:
                mask.unsafe_store(i + num_diag, UInt8(1))
            elif r + k > c:
                mask.unsafe_store(i + r, UInt8(1))
            elif r + k < c:
                mask.unsafe_store(i + r + 1, UInt8(1))


def subgraph_count(
    idx: CIPtr, rowptr: CIPtr, col: CIPtr, assoc: CIPtr, length_out: IPtr,
    num_idx: Int,
):
    """Counting half of `subgraph_cpu`.

    Upstream appends to three `std::vector`s and reports the length at the
    end; the ABI cannot allocate, so the traversal runs twice, once here and
    once in `subgraph_fill`. The loop body is upstream's unchanged.
    """
    var length = 0
    for v_new in range(num_idx):
        var v = Int(idx.unsafe_load(v_new))
        var row_start = Int(rowptr.unsafe_load(v))
        var row_end = Int(rowptr.unsafe_load(v + 1))

        for j in range(row_start, row_end):
            var w = Int(col.unsafe_load(j))
            var w_new = Int(assoc.unsafe_load(w))
            if w_new > -1:
                length += 1
    length_out.unsafe_store(0, Int64(length))


def subgraph_fill(
    idx: CIPtr, rowptr: CIPtr, col: CIPtr, assoc: CIPtr,
    out_row: IPtr, out_col: IPtr, out_idx: IPtr, num_idx: Int,
):
    """Writing half of `subgraph_cpu`."""
    var length = 0
    for v_new in range(num_idx):
        var v = Int(idx.unsafe_load(v_new))
        var row_start = Int(rowptr.unsafe_load(v))
        var row_end = Int(rowptr.unsafe_load(v + 1))

        for j in range(row_start, row_end):
            var w = Int(col.unsafe_load(j))
            var w_new = Int(assoc.unsafe_load(w))
            if w_new > -1:
                out_row.unsafe_store(length, Int64(v_new))
                out_col.unsafe_store(length, Int64(w_new))
                out_idx.unsafe_store(length, Int64(j))
                length += 1


# ====================================================== spspmm (matmul)
def spspmm_count_cpu(rowptr_a: CIPtr, col_a: CIPtr, rowptr_b: CIPtr, count_out: IPtr, num_rows: Int):
    """First pass of the SpGEMM: per-row nnz of `A @ batch`."""
    for i in range(num_rows):
        var total = Int64(0)
        var row_start = Int(rowptr_a.unsafe_load(i))
        var row_end = Int(rowptr_a.unsafe_load(i + 1))
        for e in range(row_start, row_end):
            var j = Int(col_a.unsafe_load(e))
            total += rowptr_b.unsafe_load(j + 1) - rowptr_b.unsafe_load(j)
        count_out.unsafe_store(i, total)


def spspmm_fill_cpu[dtype: DType, has_value: Bool](
    rowptr_a: CIPtr, col_a: CIPtr, value_a: CPtr[dtype], rowptr_b: CIPtr,
    col_b: CIPtr, value_b: CPtr[dtype], out_rowptr: IPtr, out_col: IPtr,
    out_value: Ptr[dtype], num_rows: Int,
):
    """Second pass of the SpGEMM: one unsorted candidate per (a, b) pair, with
    the output cursor threaded through `out_rowptr`."""
    for i in range(num_rows):
        var cursor = Int(out_rowptr.unsafe_load(i))
        var row_start = Int(rowptr_a.unsafe_load(i))
        var row_end = Int(rowptr_a.unsafe_load(i + 1))
        for e in range(row_start, row_end):
            var j = Int(col_a.unsafe_load(e))
            var va = Scalar[dtype](1)
            if has_value:
                va = value_a.unsafe_load(e)
            var inner_start = Int(rowptr_b.unsafe_load(j))
            var inner_end = Int(rowptr_b.unsafe_load(j + 1))
            for f in range(inner_start, inner_end):
                out_col.unsafe_store(cursor, col_b.unsafe_load(f))
                if has_value:
                    out_value.unsafe_store(cursor, va * value_b.unsafe_load(f))
                else:
                    out_value.unsafe_store(cursor, Scalar[dtype](1))
                cursor += 1
        out_rowptr.unsafe_store(i + 1, Int64(cursor))
