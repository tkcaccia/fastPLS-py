# fastPLS-py

`fastPLS-py` is the Python interface to the same MIT-licensed C++17 numerical
core used by the [`fastPLS`](https://github.com/tkcaccia/fastPLS) R package.
It provides NumPy-compatible float32 and float64 fitting and prediction for
the fastPLS SIMPLS-family estimator, PLS-SVD, OPLS, and linear, radial-basis,
or polynomial kernel PLS. The public `method="simpls"` name may use a bounded
candidate block; that route is not unqualified classical de Jong SIMPLS.
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

For classification, set `classifier="argmax"` or `classifier="lda"`. Numeric
responses are treated as regression unless a classifier is requested explicitly.
Calling
`predict(X, top=5)` returns the five ranked class labels for each sample.
Training scores are intentionally omitted from the compact default model; set
`store_scores=True` when they are needed for downstream analysis.

Component selection and nested validation use the same four model families:

```python
from fastpls import pls_single_cv, pls_double_cv

selected = pls_single_cv(
    X, labels, ncomp=[1, 2, 3, 4], classifier="lda",
    selection="balanced_accuracy", kfold=5, seed=7,
)
nested = pls_double_cv(
    X, labels, ncomp=[1, 2, 3, 4], classifier="lda",
    selection="balanced_accuracy", kfold_inner=5, kfold_outer=5,
    perm_test=True, times=100, seed=7,
)
```

`constrain=` keeps observations from the same exchangeability group together
in validation and permutes equal-sized groups as intact blocks. Monte Carlo
permutation p-values use the `(extreme + 1) / (completed + 1)` correction.

## R parity

`benchmarks/compare_r.py` fits the Python and R interfaces with identical data,
controls, and seeds. On the macOS arm64 validation run, SIMPLS, PLS-SVD, OPLS,
and nonlinear kernel PLS regression predictions agreed to maximum absolute
differences below `5e-16` in float64 and `2e-7` in float32. LDA class
predictions agreed for every sample in both precisions.
The recorded
CIFAR-100 float32 SIMPLS-LDA fit plus prediction took 0.211 seconds in Python
and 0.197 seconds through R, with identical 0.8687 accuracy. The machine-readable
records are under `benchmarks/`.
See `VALIDATION.md` for the exact test scope and current limitations.

## Backend status

Version 0.2.0 validates the shared portable CPU core on float32 and float64.
Requests for CUDA or Metal fail explicitly instead of silently switching to
CPU. The accelerator adapters in the R package are platform-specific runtime
layers and are not yet part of this Python distribution.

## Related repositories

- [`fastPLS`](https://github.com/tkcaccia/fastPLS): R package and canonical
  MIT-licensed C++ core.
- [`fastPLS-extra`](https://github.com/tkcaccia/fastPLS-extra): publication
  benchmarks, validation workflows, figures, and tables.
- [`fastPLS-matlab`](https://github.com/tkcaccia/fastPLS-matlab): MATLAB
  interface to the same C++ core.

## Development

```bash
python -m pip install -e '.[test]'
pytest
python benchmarks/compare_r.py
```

## License

MIT. The vendored fastPLS C++ core and the Python binding are both MIT licensed.
