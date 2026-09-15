# fastPLS formal algebraic invariants

This Lean 4 project checks exact-real-arithmetic identities used by the fastPLS
execution paths. It covers implicit cross-covariance products, compact latent
prediction, the PLS-SVD latent normal equation, SIMPLS deflation orthogonality,
the exact cached SIMPLS Gram update, the OPLS orthogonal weight, linear and
centered kernel symmetry, centered Gram entries, and subtraction of held-out
sufficient statistics in cross-validation.

The proofs do not establish floating-point accuracy, randomized-SVD error
bounds, or equivalence between the Lean specification and compiled C++/CUDA/
Metal code.

## Check the proofs

Install Lean through `elan`, then run:

```sh
lake exe cache get
lake build
```

The pinned Lean toolchain is recorded in `lean-toolchain`; `lake-manifest.json`
records the resolved Mathlib revision used for the reported proof build. The
cache step is optional. Do not run `lake update` when reproducing the build,
because it can resolve newer dependency revisions and modify the manifest.

The same command is run by `.github/workflows/lean-formal.yml` after this
project is committed to the companion repository.
