"""Loads (and if necessary builds) the compiled Mojo library.

Every kernel allocates nothing, so every call site here owns the output and
the scratch buffers and hands their addresses down. The signature table is the
ABI contract with `src/capi.mojo`; a mismatch shows up immediately as a
`ctypes` argument error rather than as silent memory corruption.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")

# `MOJO_TORCH_SPARSE_LIB` points at an already-built shared library. Set it and
# nothing is ever compiled at import or call time.
LIB = os.environ.get("MOJO_TORCH_SPARSE_LIB") or os.path.join(
    ROOT, "dist", "libmojo-torch-sparse.so"
)

I = ctypes.c_int64
F = ctypes.c_double
P = ctypes.POINTER(ctypes.c_int8)

# Every symbol in `src/capi.mojo` takes only `Int` arguments (addresses and
# extents) and returns nothing, so one signature covers the whole ABI.
_SIGNATURES = {name: ([I] * nargs, None) for name, nargs in {
    "mts_ind2ptr": 4,
    "mts_ptr2ind": 4,
    "mts_radix_sort": 6,
    "mts_gather_csr_f64": 4,
    "mts_gather_csr_f32": 4,
    "mts_non_diag_mask": 7,
    "mts_random_walk": 7,
    "mts_saint_subgraph_count": 6,
    "mts_saint_subgraph_fill": 8,
    "mts_spspmm_count": 5,
}.items()}
for _name, _nargs in [
    (f"mts_segment_csr_{t}_{r}", 7)
    for t in ("f64", "f32")
    for r in ("sum", "mean", "min", "max")
] + [
    (f"mts_scatter_{t}_{r}", 8)
    for t in ("f64", "f32")
    for r in ("sum", "mean", "min", "max")
] + [
    (f"mts_spmm_{t}{'_nv' if nv else ''}_{r}", 12)
    for t in ("f64", "f32")
    for nv in (False, True)
    for r in ("sum", "mean", "min", "max")
] + [
    (f"mts_spmm_value_bw_{t}_{r}", 11)
    for t in ("f64", "f32")
    for r in ("sum", "mean")
] + [
    (f"mts_spspmm_fill_{t}{'_nv' if nv else ''}", 10)
    for t in ("f64", "f32")
    for nv in (False, True)
]:
    _SIGNATURES[_name] = ([I] * _nargs, None)


class BuildError(RuntimeError):
    pass


class UnsupportedDType(NotImplementedError):
    pass


DTYPE_TAG = {torch.float64: "f64", torch.float32: "f32"}


def tag(dtype: torch.dtype) -> str:
    try:
        return DTYPE_TAG[dtype]
    except KeyError:
        raise UnsupportedDType(
            f"the Mojo kernels cover float32 and float64 values, got {dtype}"
        ) from None


def mojo_command() -> list[str]:
    override = os.environ.get("MOJO_TORCH_SPARSE_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_TORCH_SPARSE_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into the shared library if it is missing or stale."""
    if os.environ.get("MOJO_TORCH_SPARSE_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not os.path.isdir(SRC):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(
            f"no Mojo sources at {SRC} and no library at {LIB}; this looks like "
            "a copied install - point MOJO_TORCH_SPARSE_LIB at a built library"
        )
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "-I", SRC,
        os.path.join(SRC, "capi.mojo"), "-o", LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def call(name: str, *args) -> None:
    getattr(lib(), name)(*args)


# --------------------------------------------------------------- scratch
def empty_i64(n: int) -> torch.Tensor:
    return torch.zeros(max(n, 1), dtype=torch.long)


def empty_f64(n: int) -> torch.Tensor:
    return torch.zeros(max(n, 1), dtype=torch.float64)


def empty_f32(n: int) -> torch.Tensor:
    return torch.zeros(max(n, 1), dtype=torch.float32)


def scratch_f(dtype: torch.dtype, n: int) -> torch.Tensor:
    if dtype == torch.float64:
        return empty_f64(n)
    if dtype == torch.float32:
        return empty_f32(n)
    raise UnsupportedDType(f"no scratch dtype for {dtype}")


def main() -> int:
    """`python -m mojo_torch_sparse._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
