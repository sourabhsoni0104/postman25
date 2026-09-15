#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-.venv/bin/python}
"$PYTHON" data/get_data.py
"$PYTHON" -m pytest -q
"$PYTHON" -m scripts.complete --device "${DEVICE:-mps}"
"$PYTHON" -m scripts.report
