// SPDX-License-Identifier: MIT
#ifndef FASTPLS_PY_NATIVE_BACKEND_HPP
#define FASTPLS_PY_NATIVE_BACKEND_HPP

#include <fastpls/core/matrix.hpp>

#include <string>
#include <vector>

namespace fastpls_py {

template<class T>
class NativeBackend {
 public:
  void gemm(fastpls::core::ConstMatrixView<T> left,
            fastpls::core::ConstMatrixView<T> right,
            bool transpose_left, bool transpose_right,
            fastpls::core::MatrixView<T> output) const;

  void self_gram(fastpls::core::ConstMatrixView<T> input,
                 bool transpose_input,
                 fastpls::core::MatrixView<T> output,
                 bool full_output) const;

  bool qr_economy(fastpls::core::ConstMatrixView<T> input,
                  fastpls::core::Matrix<T>& q) const;

  bool symmetric_eigen(fastpls::core::Matrix<T>& matrix,
                       std::vector<T>& eigenvalues) const;

  bool cholesky_solve(fastpls::core::ConstMatrixView<T> matrix,
                      fastpls::core::ConstMatrixView<T> right,
                      fastpls::core::Matrix<T>& solution) const;

  bool general_solve(fastpls::core::ConstMatrixView<T> matrix,
                     fastpls::core::ConstMatrixView<T> right,
                     fastpls::core::Matrix<T>& solution) const;

  bool svd_economy(fastpls::core::ConstMatrixView<T> input, bool left_only,
                   fastpls::core::Matrix<T>& left,
                   std::vector<T>& singular_values,
                   fastpls::core::Matrix<T>& right_transpose) const;
};

std::string backend_info();

extern template class NativeBackend<float>;
extern template class NativeBackend<double>;

}  // namespace fastpls_py

#endif

