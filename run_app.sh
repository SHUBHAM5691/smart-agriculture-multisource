#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv314/bin/activate
python -m streamlit run app.py \
  --server.headless true \
  --server.port 8504 \
  --server.fileWatcherType none
