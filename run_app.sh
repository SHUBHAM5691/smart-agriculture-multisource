#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

APP_PORT="${PORT:-8504}"
python -m streamlit run app.py \
  --server.headless true \
  --server.port "$APP_PORT" \
  --server.fileWatcherType none
