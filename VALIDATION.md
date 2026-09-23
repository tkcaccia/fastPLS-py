# Validation record for fastPLS-py 0.3.0

## Evaluated source

- Wrapper version: 0.3.0
- Shared fastPLS R core version: 0.3
- Shared core commit: `82bbc48a0d69e4bd0d7c261fdcc8e636694133b1`
- Platform: macOS arm64
- CPU linear algebra: Apple Accelerate

`tools/check_core.py` confirms that the vendored headers are identical to the
recorded R source.

## Automated tests

`python -m pytest -q` completed with 76 passed tests and one skipped optional
Matplotlib rendering test because Matplotlib was not installed. The suite covers all four
PLS families, float32 and float64, regression, argmax and LDA classification,
ranked prediction, rSVD, row- and column-wise correlation, independent-test
evaluation, single and nested cross-validation, grouped permutation testing,
VIP output, and intercept-only prediction when a SIMPLS-family fit has zero
effective directions.

The source distribution was rebuilt as a wheel successfully. A clean virtual
environment loaded version 0.3.0 from that wheel and reran the complete suite.
The additional test verifies default LDA classification and explicit
CUDA-unavailable metadata without CPU fallback.

## R interface agreement

The float64 record is in `benchmarks/parity_macos_arm64_v0.2.0.json`; the
float32 record is in `benchmarks/parity_macos_arm64_v0.2.0_float32.json`.

| Family | Float64 maximum difference | Float32 maximum difference | LDA agreement |
|---|---:|---:|---:|
| SIMPLS | 2.15e-16 | 6.71e-08 | 100% |
| PLS-SVD | 4.34e-16 | 1.98e-07 | 100% |
| OPLS | 3.19e-16 | 2.07e-07 | 100% |
| Kernel PLS | 6.59e-17 | 1.86e-09 | 100% |

These results establish agreement for the recorded arrays and controls. They
do not establish equivalence between rSVD and a full decomposition or replace
validation on other operating systems.

## Current limitation

The Python package exposes the complete portable CPU workflow. CUDA and Metal
adapters are not included in version 0.3.0; requesting either backend fails
explicitly rather than changing to CPU.
