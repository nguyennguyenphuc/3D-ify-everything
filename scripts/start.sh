#!/bin/zsh
set -eu
cd "${0:A:h:h}"
export PYTORCH_ENABLE_MPS_FALLBACK=0
export PYTHONUNBUFFERED=1
if [[ -d /Applications/Xcode.app/Contents/Developer ]]; then
  export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi
exec rtk proxy .venv/bin/python -m uvicorn studio.api:app --host 127.0.0.1 --port 8000
