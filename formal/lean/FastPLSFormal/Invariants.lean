import Mathlib.Data.Matrix.Mul
import Mathlib.Basic.Real.Basic
import Mathlib.LinearAlgebra.Matrix.DotProduct
import Mathlib.Tactic.Abel
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-!
# Algebraic invariants used by fastPLS

These theorems formalize exact-real-arithmetic identities used by the optimized
PLS-SVD, SIMPLS-family, OPLS, kernel-PLS, and cross-validation implementations.
They do not verify floating-point rounding, randomized-SVD approximation error,
or correspondence between this specification and compiled CPU/GPU code.
-/

open scoped BigOperators Matrix

namespace FastPLSFormal

set_option linter.unusedSectionVars false

section MatrixProducts

variable {n p q k a : Type*}
variable [Fintype n] [Fintype p] [Fintype q] [Fintype k] [Fintype a]
variable [DecidableEq n] [DecidableEq p] [DecidableEq q]
variable [DecidableEq k] [DecidableEq a]

/-- The matrix-free cross-covariance action equals explicit materialization. -/
theorem implicit_crossCovariance
    (X : Matrix n p ℝ) (Y : Matrix n q ℝ) (Omega : Matrix q k ℝ) :
    (X.transpose * (Y * Omega : Matrix n k ℝ) : Matrix p k ℝ) =
      ((X.transpose * Y : Matrix p q ℝ) * Omega : Matrix p k ℝ) := by
  rw [Matrix.mul_assoc]

/-- Compact latent prediction equals prediction with the dense coefficient matrix. -/
theorem compact_prediction
    (Xnew : Matrix n p ℝ) (R : Matrix p a ℝ) (Q : Matrix q a ℝ) :
    ((Xnew * R : Matrix n a ℝ) * Q.transpose : Matrix n q ℝ) =
      (Xnew * (R * Q.transpose : Matrix p q ℝ) : Matrix n q ℝ) := by
  rw [Matrix.mul_assoc]

/-- The PLS-SVD latent solve satisfies the corresponding normal equation. -/
theorem plssvd_latent_normal_equation
    (H L D : Matrix a a ℝ) (V : Matrix q a ℝ)
    (hsolve : (H * L : Matrix a a ℝ) = D) :
    (H * (L * V.transpose : Matrix a q ℝ) : Matrix a q ℝ) =
      (D * V.transpose : Matrix a q ℝ) := by
  rw [← Matrix.mul_assoc, hsolve]

/-- A normalized SIMPLS deflation direction is orthogonal to the deflated state. -/
theorem simpls_deflation_orthogonal
    (S : Matrix p q ℝ) (v : Matrix p (Fin 1) ℝ)
    (hv : (v.transpose * v : Matrix (Fin 1) (Fin 1) ℝ) = 1) :
    (v.transpose *
      (S - v * (v.transpose * S : Matrix (Fin 1) q ℝ) : Matrix p q ℝ) :
        Matrix (Fin 1) q ℝ) = 0 := by
  rw [Matrix.mul_sub, ← Matrix.mul_assoc, hv, Matrix.one_mul, sub_self]

/-- The rank-one update used for the cached SIMPLS Gram matrix is exact. -/
theorem simpls_cached_gram_entry
    (v x y : p → ℝ) (hv : dotProduct v v = 1) :
    dotProduct (x - (dotProduct v x) • v)
        (y - (dotProduct v y) • v) =
      dotProduct x y - dotProduct v x * dotProduct v y := by
  simp only [dotProduct_sub, dotProduct_smul, smul_eq_mul, hv,
    dotProduct_comm]
  ring

/-- A linear-kernel Gram matrix is symmetric. -/
theorem linear_kernel_symmetric (X : Matrix n p ℝ) :
    (X * X.transpose : Matrix n n ℝ).transpose =
      (X * X.transpose : Matrix n n ℝ) := by
  rw [Matrix.transpose_mul, Matrix.transpose_transpose]

/-- Symmetric double centering preserves the symmetry of a kernel matrix. -/
theorem double_centering_preserves_symmetry
    (H K : Matrix n n ℝ) (hH : H.transpose = H) (hK : K.transpose = K) :
    (H * K * H : Matrix n n ℝ).transpose =
      (H * K * H : Matrix n n ℝ) := by
  rw [Matrix.transpose_mul, Matrix.transpose_mul, hH, hK, Matrix.mul_assoc]

end MatrixProducts

section VectorIdentities

variable {i : Type*} [Fintype i]

/-- OPLS removes the component of `p` parallel to a nonzero predictive weight. -/
theorem opls_weight_is_orthogonal
    (w p : i → ℝ) (hww : dotProduct w w ≠ 0) :
    dotProduct w (p - (dotProduct w p / dotProduct w w) • w) = 0 := by
  rw [dotProduct_sub, dotProduct_smul]
  simp only [smul_eq_mul]
  field_simp
  ring

/-- Pairwise centering can be applied as a correction to an uncentered Gram entry. -/
theorem centered_gram_entry
    (x y mu : i → ℝ) :
    dotProduct (x - mu) (y - mu) =
      dotProduct x y - dotProduct x mu - dotProduct mu y + dotProduct mu mu := by
  simp only [dotProduct, Pi.sub_apply]
  simp_rw [sub_mul, mul_sub]
  simp_rw [Finset.sum_sub_distrib]
  ring

end VectorIdentities

section CrossValidation

variable {i M : Type*} [DecidableEq i] [AddCommGroup M]

/-- A training-fold sufficient statistic is the full statistic minus holdout data. -/
theorem training_statistic_by_subtraction
    (train holdout : Finset i) (f : i → M)
    (hdisjoint : Disjoint train holdout) :
    (∑ j ∈ train ∪ holdout, f j) - (∑ j ∈ holdout, f j) =
      ∑ j ∈ train, f j := by
  rw [Finset.sum_union hdisjoint]
  abel

end CrossValidation

end FastPLSFormal
