#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Removing JAX packages that conflict with TensorFlow 2.16 ml-dtypes"
python -m pip uninstall -y jax jaxlib || true

echo "==> Installing Colab requirements"
python -m pip install -r "$ROOT_DIR/requirements-colab.txt"

echo "==> Dependency check"
python - <<'PY'
import tensorflow as tf
import numpy as np
import scipy
import sklearn

print("tensorflow", tf.__version__)
print("numpy", np.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
PY
