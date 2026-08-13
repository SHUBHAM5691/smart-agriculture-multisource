"""Run LightGBM outside Streamlit's native-library process."""

import json
import os
import subprocess
import sys


def predict_fertilizer(inputs: dict) -> dict:
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    completed = subprocess.run(
        [sys.executable, "-m", "prediction.fertilizer_worker"],
        input=json.dumps(inputs),
        text=True,
        capture_output=True,
        timeout=60,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        message = detail[-1] if detail else f"worker exited with code {completed.returncode}"
        raise RuntimeError(f"Fertilizer prediction worker failed: {message}")
    return json.loads(completed.stdout)
