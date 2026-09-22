#!/bin/zsh
set -eu
cd "${0:A:h:h}"
rtk proxy uv sync --frozen
rtk npm ci --prefix web
rtk proxy .venv/bin/python scripts/fetch_upstream.py
rtk proxy .venv/bin/python scripts/download_assets.py
rtk proxy .venv/bin/python scripts/fix_libtorch_openmp.py
rtk proxy .venv/bin/python scripts/build_opensplat.py
rtk npm run build --prefix web
