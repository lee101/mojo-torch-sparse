"""Build upstream `torch_sparse` from its PyPI sdist, for parity testing.

Upstream ships no wheel on PyPI (only source distributions) and conda-forge
carries no `torch-sparse` package, so the extension has to be compiled. It is
built with CUDA disabled and no METIS, which is exactly the surface this port
mirrors. The result is optional: `tests/conftest.py` enables the extra
`torch_sparse` comparisons only when `import torch_sparse` succeeds.

Run with `pixi run install-upstream`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

SDIST = "torch-sparse==0.6.18"


def main() -> int:
    try:
        import torch_sparse  # noqa: F401
    except ImportError:
        pass
    else:
        print(f"torch_sparse already installed: {torch_sparse.__file__}")
        return 0

    env = dict(os.environ)
    env.update({
        "MAX_JOBS": env.get("MAX_JOBS", "8"),
        "CXX": "g++",
        "TORCH_CUDA_ARCH_LIST": "",
    })
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--no-build-isolation", "--no-deps",
            "--cache-dir", os.path.join(tmp, "cache"),
            SDIST,
        ]
        print("$", " ".join(cmd), flush=True)
        return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())