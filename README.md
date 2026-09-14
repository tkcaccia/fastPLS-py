# fastPLS-py

`fastPLS-py` is the Python interface to the same MIT-licensed C++17 numerical
core used by the [`fastPLS`](https://github.com/tkcaccia/fastPLS) R package.
It provides NumPy-compatible float32 and float64 fitting and prediction for
SIMPLS, PLS-SVD, OPLS, and linear, radial-basis, or polynomial kernel PLS.
Classification supports argmax and pooled-covariance LDA prediction heads.

The vendored core is pinned in `UPSTREAM_CORE.json`; `tools/check_core.py`
verifies that it is byte-for-byte identical to the recorded R-package source.

## Install

```bash
python -m pip install .
```

Building requires a C++17 compiler, CMake, pybind11, NumPy, and BLAS/LAPACK.
Apple builds link Accelerate. Linux and Windows builds use the system BLAS and
LAPACK selected by CMake; OpenBLAS is recommended.

## Use

```python
import numpy as np
from fastpls import PLS, evaluate

rng = np.random.default_rng(7)
X = np.asfortranarray(rng.normal(size=(200, 30)).astype(np.float32))
y = X[:, 0] - 0.5 * X[:, 1]

model = PLS(n_components=3, method="simpls", seed=7).fit(X[:150], y[:150])
prediction = model.predict(X[150:])
print(evaluate(y[150:], prediction[:, 0]))
```

For classification, set `classifier="argmax"` or `classifier="lda"`. Calling
`predict(X, top=5)` returns the five ranked class labels for each sample.
Training scores are intentionally omitted from the compact default model; set
`store_scores=True` when they are needed for downstream analysis.

## R parity

`benchmarks/compare_r.py` fits the Python and R interfaces with identical data,
controls, and seeds. On the macOS arm64 validation run, SIMPLS and PLS-SVD
regression predictions agreed to maximum absolute differences below
`6e-16`, and LDA class predictions agreed for every sample. The recorded
CIFAR-100 float32 SIMPLS-LDA fit plus prediction took 0.211 seconds in Python
and 0.197 seconds through R, with identical 0.8687 accuracy. The machine-readable
records are under `benchmarks/`.

## Backend status

Version 0.1.0 validates the shared portable CPU core on float32 and float64.
Requests for CUDA or Metal fail explicitly instead of silently switching to
CPU. The accelerator adapters in the R package are platform-specific runtime
layers and are not yet part of this Python distribution.

## Development

```bash
python -m pip install -e '.[test]'
pytest
python benchmarks/compare_r.py
```

## License

MIT. The vendored fastPLS C++ core and the Python binding are both MIT licensed.
