#!/usr/bin/env bash
# Launcher for scripts/show_depth.py
#   Interactive depth viewer: hover the mouse over the map to read the depth
#   value and the back-projected camera-frame XYZ of that pixel.
#
# Notes:
#   - Default depth file is assets/imgs/zerith_ref_depth.npy (float32, metres);
#     override it with e.g. --npy <path> [--scale auto|0.001|1.0].
#   - A matplotlib window is required, so run this in a display-enabled
#     environment.
#   - Extra arguments are forwarded to show_depth.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT"

/home/jszn/miniconda3/envs/foundationpose/bin/python \
    scripts/show_depth.py "$@"
