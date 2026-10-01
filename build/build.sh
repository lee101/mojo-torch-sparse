#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"

# src/capi.mojo is the single entry point; -I src resolves the `ported`
# module that holds the kernels.
# -j caps compiler threads: the shared CI host runs many jobs and the default
# (all cores) fails with pthread_create EAGAIN under load.
mojo build --emit shared-lib -j "${MOJO_BUILD_JOBS:-4}" -I "$repo_dir/src" \
    "$repo_dir/src/capi.mojo" \
    -o "$repo_dir/dist/libmojo-torch-sparse.so"
