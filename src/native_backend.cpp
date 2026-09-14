// SPDX-License-Identifier: MIT
#include "native_backend.hpp"

#include <fastpls/core/linalg.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(FASTPLS_PY_ACCELERATE)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#elif __has_include(<cblas.h>)
#include <cblas.h>
#define FASTPLS_PY_HAS_CBLAS 1
#endif

extern "C" {
void sgeqrf_(const int*, const int*, float*, const int*, float*, float*,
             const int*, int*);
void dgeqrf_(const int*, const int*, double*, const int*, double*, double*,
             const int*, int*);
void sorgqr_(const int*, const int*, const int*, float*, const int*,
             const float*, float*, const int*, int*);
void dorgqr_(const int*, const int*, const int*, double*, const int*,
             const double*, double*, const int*, int*);
void ssyevd_(const char*, const char*, const int*, float*, const int*, float*,
             float*, const int*, int*, const int*, int*);
void dsyevd_(const char*, const char*, const int*, double*, const int*, double*,
             double*, const int*, int*, const int*, int*);
void spotrf_(const char*, const int*, float*, const int*, int*);
void dpotrf_(const char*, const int*, double*, const int*, int*);
void spotrs_(const char*, const int*, const int*, const float*, const int*,
             float*, const int*, int*);
void dpotrs_(const char*, const int*, const int*, const double*, const int*,
             double*, const int*, int*);
void sgesv_(const int*, const int*, float*, const int*, int*, float*,
            const int*, int*);
void dgesv_(const int*, const int*, double*, const int*, int*, double*,
            const int*, int*);
void sgesdd_(const char*, const int*, const int*, float*, const int*, float*,
             float*, const int*, float*, const int*, float*, const int*, int*,
             int*);
void dgesdd_(const char*, const int*, const int*, double*, const int*, double*,
             double*, const int*, double*, const int*, double*, const int*, int*,
             int*);
}

namespace fastpls_py {
namespace {

int checked_int(std::size_t value, const char* operation) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(std::string(operation) + " dimension is too large");
  }
  return static_cast<int>(value);
}

template<class T>
fastpls::core::Matrix<T> contiguous_copy(
    fastpls::core::ConstMatrixView<T> input) {
  fastpls::core::Matrix<T> output(input.rows(), input.columns());
  for (std::size_t column = 0; column < input.columns(); ++column) {
    std::copy_n(input.data() + column * input.leading_dimension(), input.rows(),
                output.data() + column * output.rows());
  }
  return output;
}

template<class T>
int workspace_size(T value) {
  if (!std::isfinite(value) || value < T(1) ||
      value > static_cast<T>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("invalid LAPACK workspace query");
  }
  return static_cast<int>(value);
}

template<class T>
void mirror_lower(fastpls::core::MatrixView<T> matrix) {
  for (std::size_t column = 0; column < matrix.columns(); ++column) {
    for (std::size_t row = 0; row < column; ++row) {
      matrix(row, column) = matrix(column, row);
    }
  }
}

template<class T>
void reference_gemm(fastpls::core::ConstMatrixView<T> left,
                    fastpls::core::ConstMatrixView<T> right,
                    bool transpose_left, bool transpose_right,
                    fastpls::core::MatrixView<T> output) {
  fastpls::core::reference_gemm(
      left, right, transpose_left, transpose_right, output);
}

template<class T>
void blas_gemm(fastpls::core::ConstMatrixView<T> left,
               fastpls::core::ConstMatrixView<T> right,
               bool transpose_left, bool transpose_right,
               fastpls::core::MatrixView<T> output) {
#if defined(FASTPLS_PY_ACCELERATE) || defined(FASTPLS_PY_HAS_CBLAS)
  const int m = checked_int(transpose_left ? left.columns() : left.rows(), "GEMM");
  const int n = checked_int(transpose_right ? right.rows() : right.columns(), "GEMM");
  const int k = checked_int(transpose_left ? left.rows() : left.columns(), "GEMM");
  const int lda = checked_int(left.leading_dimension(), "GEMM");
  const int ldb = checked_int(right.leading_dimension(), "GEMM");
  const int ldc = checked_int(output.leading_dimension(), "GEMM");
  const CBLAS_TRANSPOSE ta = transpose_left ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE tb = transpose_right ? CblasTrans : CblasNoTrans;
  if constexpr (std::is_same<T, float>::value) {
    cblas_sgemm(CblasColMajor, ta, tb, m, n, k, 1.0f, left.data(), lda,
                right.data(), ldb, 0.0f, output.data(), ldc);
  } else {
    cblas_dgemm(CblasColMajor, ta, tb, m, n, k, 1.0, left.data(), lda,
                right.data(), ldb, 0.0, output.data(), ldc);
  }
#else
  reference_gemm(left, right, transpose_left, transpose_right, output);
#endif
}

template<class T>
bool lapack_qr(fastpls::core::ConstMatrixView<T> input,
               fastpls::core::Matrix<T>& q) {
  const int m = checked_int(input.rows(), "QR");
  const int n = checked_int(input.columns(), "QR");
  const int rank = std::min(m, n);
  if (rank == 0) {
    q.resize(input.rows(), 0);
    return true;
  }
  const int lda = std::max(1, m);
  auto factor = contiguous_copy(input);
  std::vector<T> tau(static_cast<std::size_t>(rank));
  int info = 0;
  int lwork = -1;
  T query = T(0);
  if constexpr (std::is_same<T, float>::value) {
    sgeqrf_(&m, &n, factor.data(), &lda, tau.data(), &query, &lwork, &info);
  } else {
    dgeqrf_(&m, &n, factor.data(), &lda, tau.data(), &query, &lwork, &info);
  }
  if (info != 0) return false;
  lwork = workspace_size(query);
  std::vector<T> work(static_cast<std::size_t>(lwork));
  if constexpr (std::is_same<T, float>::value) {
    sgeqrf_(&m, &n, factor.data(), &lda, tau.data(), work.data(), &lwork, &info);
  } else {
    dgeqrf_(&m, &n, factor.data(), &lda, tau.data(), work.data(), &lwork, &info);
  }
  if (info != 0) return false;
  q.resize(input.rows(), static_cast<std::size_t>(rank));
  for (int column = 0; column < rank; ++column) {
    std::copy_n(factor.data() + static_cast<std::size_t>(column) * factor.rows(),
                input.rows(), q.data() + static_cast<std::size_t>(column) * q.rows());
  }
  lwork = -1;
  query = T(0);
  if constexpr (std::is_same<T, float>::value) {
    sorgqr_(&m, &rank, &rank, q.data(), &lda, tau.data(), &query, &lwork, &info);
  } else {
    dorgqr_(&m, &rank, &rank, q.data(), &lda, tau.data(), &query, &lwork, &info);
  }
  if (info != 0) return false;
  lwork = workspace_size(query);
  work.resize(static_cast<std::size_t>(lwork));
  if constexpr (std::is_same<T, float>::value) {
    sorgqr_(&m, &rank, &rank, q.data(), &lda, tau.data(), work.data(), &lwork, &info);
  } else {
    dorgqr_(&m, &rank, &rank, q.data(), &lda, tau.data(), work.data(), &lwork, &info);
  }
  return info == 0;
}

template<class T>
bool lapack_eigen(fastpls::core::Matrix<T>& matrix,
                  std::vector<T>& eigenvalues) {
  if (matrix.rows() != matrix.columns()) return false;
  const int n = checked_int(matrix.rows(), "eigendecomposition");
  if (n == 0) {
    eigenvalues.clear();
    return true;
  }
  const int lda = std::max(1, n);
  eigenvalues.resize(static_cast<std::size_t>(n));
  const char vectors = 'V';
  const char lower = 'L';
  int info = 0;
  int lwork = -1;
  int liwork = -1;
  T query = T(0);
  int iquery = 0;
  if constexpr (std::is_same<T, float>::value) {
    ssyevd_(&vectors, &lower, &n, matrix.data(), &lda, eigenvalues.data(),
            &query, &lwork, &iquery, &liwork, &info);
  } else {
    dsyevd_(&vectors, &lower, &n, matrix.data(), &lda, eigenvalues.data(),
            &query, &lwork, &iquery, &liwork, &info);
  }
  if (info != 0) return false;
  lwork = workspace_size(query);
  liwork = std::max(1, iquery);
  std::vector<T> work(static_cast<std::size_t>(lwork));
  std::vector<int> iwork(static_cast<std::size_t>(liwork));
  if constexpr (std::is_same<T, float>::value) {
    ssyevd_(&vectors, &lower, &n, matrix.data(), &lda, eigenvalues.data(),
            work.data(), &lwork, iwork.data(), &liwork, &info);
  } else {
    dsyevd_(&vectors, &lower, &n, matrix.data(), &lda, eigenvalues.data(),
            work.data(), &lwork, iwork.data(), &liwork, &info);
  }
  return info == 0;
}

template<class T>
bool lapack_cholesky(fastpls::core::ConstMatrixView<T> matrix,
                     fastpls::core::ConstMatrixView<T> right,
                     fastpls::core::Matrix<T>& solution) {
  if (matrix.rows() != matrix.columns() || matrix.rows() != right.rows()) return false;
  const int n = checked_int(matrix.rows(), "Cholesky solve");
  const int nrhs = checked_int(right.columns(), "Cholesky solve");
  const int lda = std::max(1, n);
  const int ldb = std::max(1, n);
  auto factor = contiguous_copy(matrix);
  solution = contiguous_copy(right);
  const char lower = 'L';
  int info = 0;
  if constexpr (std::is_same<T, float>::value) {
    spotrf_(&lower, &n, factor.data(), &lda, &info);
    if (info == 0) spotrs_(&lower, &n, &nrhs, factor.data(), &lda,
                           solution.data(), &ldb, &info);
  } else {
    dpotrf_(&lower, &n, factor.data(), &lda, &info);
    if (info == 0) dpotrs_(&lower, &n, &nrhs, factor.data(), &lda,
                           solution.data(), &ldb, &info);
  }
  return info == 0;
}

template<class T>
bool lapack_solve(fastpls::core::ConstMatrixView<T> matrix,
                  fastpls::core::ConstMatrixView<T> right,
                  fastpls::core::Matrix<T>& solution) {
  if (matrix.rows() != matrix.columns() || matrix.rows() != right.rows()) return false;
  const int n = checked_int(matrix.rows(), "linear solve");
  const int nrhs = checked_int(right.columns(), "linear solve");
  const int lda = std::max(1, n);
  const int ldb = std::max(1, n);
  auto factor = contiguous_copy(matrix);
  solution = contiguous_copy(right);
  std::vector<int> pivots(static_cast<std::size_t>(n));
  int info = 0;
  if constexpr (std::is_same<T, float>::value) {
    sgesv_(&n, &nrhs, factor.data(), &lda, pivots.data(), solution.data(), &ldb, &info);
  } else {
    dgesv_(&n, &nrhs, factor.data(), &lda, pivots.data(), solution.data(), &ldb, &info);
  }
  return info == 0;
}

template<class T>
bool lapack_svd(fastpls::core::ConstMatrixView<T> input, bool left_only,
                fastpls::core::Matrix<T>& left,
                std::vector<T>& singular_values,
                fastpls::core::Matrix<T>& right_transpose) {
  const int m = checked_int(input.rows(), "SVD");
  const int n = checked_int(input.columns(), "SVD");
  const int rank = std::min(m, n);
  if (rank == 0) return false;
  const int lda = std::max(1, m);
  const int ldu = std::max(1, m);
  const int ldvt = std::max(1, rank);
  auto factor = contiguous_copy(input);
  left.resize(input.rows(), static_cast<std::size_t>(rank));
  right_transpose.resize(static_cast<std::size_t>(rank), input.columns());
  singular_values.resize(static_cast<std::size_t>(rank));
  std::vector<int> iwork(static_cast<std::size_t>(8 * rank));
  const char job = 'S';
  int info = 0;
  int lwork = -1;
  T query = T(0);
  if constexpr (std::is_same<T, float>::value) {
    sgesdd_(&job, &m, &n, factor.data(), &lda, singular_values.data(),
            left.data(), &ldu, right_transpose.data(), &ldvt, &query, &lwork,
            iwork.data(), &info);
  } else {
    dgesdd_(&job, &m, &n, factor.data(), &lda, singular_values.data(),
            left.data(), &ldu, right_transpose.data(), &ldvt, &query, &lwork,
            iwork.data(), &info);
  }
  if (info != 0) return false;
  lwork = workspace_size(query);
  std::vector<T> work(static_cast<std::size_t>(lwork));
  factor = contiguous_copy(input);
  if constexpr (std::is_same<T, float>::value) {
    sgesdd_(&job, &m, &n, factor.data(), &lda, singular_values.data(),
            left.data(), &ldu, right_transpose.data(), &ldvt, work.data(),
            &lwork, iwork.data(), &info);
  } else {
    dgesdd_(&job, &m, &n, factor.data(), &lda, singular_values.data(),
            left.data(), &ldu, right_transpose.data(), &ldvt, work.data(),
            &lwork, iwork.data(), &info);
  }
  if (left_only) right_transpose = fastpls::core::Matrix<T>();
  return info == 0;
}

}  // namespace

template<class T>
void NativeBackend<T>::gemm(fastpls::core::ConstMatrixView<T> left,
                            fastpls::core::ConstMatrixView<T> right,
                            bool transpose_left, bool transpose_right,
                            fastpls::core::MatrixView<T> output) const {
  blas_gemm(left, right, transpose_left, transpose_right, output);
}

template<class T>
void NativeBackend<T>::self_gram(fastpls::core::ConstMatrixView<T> input,
                                 bool transpose_input,
                                 fastpls::core::MatrixView<T> output,
                                 bool full_output) const {
#if defined(FASTPLS_PY_ACCELERATE) || defined(FASTPLS_PY_HAS_CBLAS)
  const int dimension = checked_int(
      transpose_input ? input.columns() : input.rows(), "SYRK");
  const int rank = checked_int(
      transpose_input ? input.rows() : input.columns(), "SYRK");
  const int lda = checked_int(input.leading_dimension(), "SYRK");
  const int ldc = checked_int(output.leading_dimension(), "SYRK");
  const CBLAS_TRANSPOSE operation =
      transpose_input ? CblasTrans : CblasNoTrans;
  if constexpr (std::is_same<T, float>::value) {
    cblas_ssyrk(CblasColMajor, CblasLower, operation, dimension, rank, 1.0f,
                input.data(), lda, 0.0f, output.data(), ldc);
  } else {
    cblas_dsyrk(CblasColMajor, CblasLower, operation, dimension, rank, 1.0,
                input.data(), lda, 0.0, output.data(), ldc);
  }
#else
  blas_gemm(input, input, transpose_input, !transpose_input, output);
#endif
  if (full_output) mirror_lower(output);
}

template<class T>
bool NativeBackend<T>::qr_economy(fastpls::core::ConstMatrixView<T> input,
                                  fastpls::core::Matrix<T>& q) const {
  return lapack_qr(input, q);
}

template<class T>
bool NativeBackend<T>::symmetric_eigen(fastpls::core::Matrix<T>& matrix,
                                       std::vector<T>& eigenvalues) const {
  return lapack_eigen(matrix, eigenvalues);
}

template<class T>
bool NativeBackend<T>::cholesky_solve(
    fastpls::core::ConstMatrixView<T> matrix,
    fastpls::core::ConstMatrixView<T> right,
    fastpls::core::Matrix<T>& solution) const {
  return lapack_cholesky(matrix, right, solution);
}

template<class T>
bool NativeBackend<T>::general_solve(
    fastpls::core::ConstMatrixView<T> matrix,
    fastpls::core::ConstMatrixView<T> right,
    fastpls::core::Matrix<T>& solution) const {
  return lapack_solve(matrix, right, solution);
}

template<class T>
bool NativeBackend<T>::svd_economy(
    fastpls::core::ConstMatrixView<T> input, bool left_only,
    fastpls::core::Matrix<T>& left, std::vector<T>& singular_values,
    fastpls::core::Matrix<T>& right_transpose) const {
  return lapack_svd(input, left_only, left, singular_values, right_transpose);
}

std::string backend_info() {
#if defined(FASTPLS_PY_ACCELERATE)
  return "Apple Accelerate";
#elif defined(FASTPLS_PY_HAS_CBLAS)
  return "system CBLAS/LAPACK";
#else
  return "portable GEMM with system LAPACK";
#endif
}

template class NativeBackend<float>;
template class NativeBackend<double>;

}  // namespace fastpls_py
