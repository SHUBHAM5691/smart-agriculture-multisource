"""Clean-process entry point for the native LightGBM fertilizer model."""

import json
import sys

from prediction.real_predictors import RealFertilizerModel


def main() -> int:
    inputs = json.load(sys.stdin)
    result = RealFertilizerModel().predict(inputs)
    json.dump(result, sys.stdout, allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
