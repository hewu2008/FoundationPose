#!/usr/bin/env bash
# Launcher for policy/01_minimal_color.py
#   Minimal color viewer + semi-transparent image calibration overlay
#
# Notes:
#   - GRPC_TARGET / CAMERA_NAME / CALIB_IMAGE are configured inside policy/01_minimal_color.py.
#   - camera_client is the compiled extension policy/camera_client.cpython-310-*.so in the
#     same directory; when the script runs directly that directory is already on sys.path,
#     so no extra PYTHONPATH is needed.
#   - That extension is built for CPython 3.10, so use Python 3.10 to run this script.
#   - An OpenCV window is required, so run this in a display-enabled environment.
set -euo pipefail

python policy/01_minimal_color.py "$@"
