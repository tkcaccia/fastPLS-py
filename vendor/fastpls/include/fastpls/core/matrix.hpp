// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_MATRIX_HPP
#define FASTPLS_CORE_MATRIX_HPP

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace fastpls {
namespace core {

// Page-aligned storage lets Apple-silicon builds expose large matrices to
// Metal as shared buffers without a staging copy. The same allocator remains
// portable on CPU-only and CUDA builds.
template<class T>
class MatrixAllocator {
 public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using propagate_on_container_move_assignment = std::true_type;
  using is_always_equal = std::true_type;

  MatrixAllocator() noexcept = default;

  template<class U>
  MatrixAllocator(const MatrixAllocator<U>&) noexcept {}

  T* allocate(std::size_t count) {
    if (count == 0) return nullptr;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }
    constexpr std::size_t alignment = 16384;
    const std::size_t bytes = count * sizeof(T);
    if (bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
      throw std::bad_array_new_length();
    }
    const std::size_t allocated =
      ((bytes + alignment - 1) / alignment) * alignment;
#if defined(_WIN32)
    void* memory = _aligned_malloc(allocated, alignment);
    if (memory == nullptr) throw std::bad_alloc();
#else
    void* memory = nullptr;
    if (posix_memalign(&memory, alignment, allocated) != 0) {
      throw std::bad_alloc();
    }
#endif
    return static_cast<T*>(memory);
  }

  void deallocate(T* pointer, std::size_t) noexcept {
#if defined(_WIN32)
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
  }
};

template<class T, class U>
bool operator==(const MatrixAllocator<T>&, const MatrixAllocator<U>&) noexcept {
  return true;
}

template<class T, class U>
bool operator!=(const MatrixAllocator<T>&, const MatrixAllocator<U>&) noexcept {
  return false;
}

template<class T>
class BasicMatrixView {
 public:
  using value_type = typename std::remove_const<T>::type;
  using pointer = T*;
  using reference = T&;

  constexpr BasicMatrixView() noexcept = default;

  constexpr BasicMatrixView(pointer data, std::size_t rows,
                            std::size_t columns,
                            std::size_t leading_dimension) noexcept
      : data_(data), rows_(rows), columns_(columns),
        leading_dimension_(leading_dimension) {}

  template<class U,
           typename std::enable_if<
             std::is_const<T>::value &&
             std::is_same<U, value_type>::value,
             int
           >::type = 0>
  constexpr BasicMatrixView(const BasicMatrixView<U>& other) noexcept
      : data_(other.data()), rows_(other.rows()), columns_(other.columns()),
        leading_dimension_(other.leading_dimension()) {}

  constexpr pointer data() const noexcept { return data_; }
  constexpr std::size_t rows() const noexcept { return rows_; }
  constexpr std::size_t columns() const noexcept { return columns_; }
  constexpr std::size_t leading_dimension() const noexcept {
    return leading_dimension_;
  }
  constexpr bool empty() const noexcept {
    return rows_ == 0 || columns_ == 0;
  }
  constexpr bool contiguous() const noexcept {
    return leading_dimension_ == rows_;
  }

  reference operator()(std::size_t row, std::size_t column) const {
    if (row >= rows_ || column >= columns_) {
      throw std::out_of_range("fastPLS matrix-view index is out of range");
    }
    return data_[row + column * leading_dimension_];
  }

 private:
  pointer data_ = nullptr;
  std::size_t rows_ = 0;
  std::size_t columns_ = 0;
  std::size_t leading_dimension_ = 0;
};

template<class T>
using MatrixView = BasicMatrixView<T>;

template<class T>
using ConstMatrixView = BasicMatrixView<const T>;

template<class T>
class Matrix {
 public:
  Matrix() = default;

  Matrix(std::size_t rows, std::size_t columns)
      : values_(rows * columns), rows_(rows), columns_(columns) {}

  void resize(std::size_t rows, std::size_t columns) {
    values_.resize(rows * columns);
    rows_ = rows;
    columns_ = columns;
  }

  T* data() noexcept { return values_.data(); }
  const T* data() const noexcept { return values_.data(); }
  std::size_t rows() const noexcept { return rows_; }
  std::size_t columns() const noexcept { return columns_; }
  std::size_t size() const noexcept { return values_.size(); }

  MatrixView<T> view() noexcept {
    return MatrixView<T>(data(), rows_, columns_, rows_);
  }

  ConstMatrixView<T> view() const noexcept {
    return ConstMatrixView<T>(data(), rows_, columns_, rows_);
  }

  T& operator()(std::size_t row, std::size_t column) {
    return view()(row, column);
  }

  const T& operator()(std::size_t row, std::size_t column) const {
    return view()(row, column);
  }

 private:
  std::vector<T, MatrixAllocator<T>> values_;
  std::size_t rows_ = 0;
  std::size_t columns_ = 0;
};

// Column padding is useful when a matrix is already being gathered into owned
// storage and will be consumed repeatedly by level-3 kernels. It is kept as a
// separate type so ordinary Matrix code can continue to rely on contiguous
// rows * columns storage.
template<class T>
class PaddedMatrix {
 public:
  PaddedMatrix() = default;

  void resize(std::size_t rows, std::size_t columns,
              std::size_t row_multiple) {
    if (row_multiple == 0 ||
        rows > std::numeric_limits<std::size_t>::max() -
          (row_multiple - 1)) {
      throw std::invalid_argument(
        "fastPLS padded-matrix dimensions are invalid"
      );
    }
    const std::size_t leading =
      ((rows + row_multiple - 1) / row_multiple) * row_multiple;
    if (columns != 0 &&
        leading > std::numeric_limits<std::size_t>::max() / columns) {
      throw std::bad_array_new_length();
    }
    values_.resize(leading * columns);
    rows_ = rows;
    columns_ = columns;
    leading_dimension_ = leading;
  }

  T* data() noexcept { return values_.data(); }
  const T* data() const noexcept { return values_.data(); }
  std::size_t rows() const noexcept { return rows_; }
  std::size_t columns() const noexcept { return columns_; }
  std::size_t leading_dimension() const noexcept {
    return leading_dimension_;
  }

  MatrixView<T> view() noexcept {
    return MatrixView<T>(data(), rows_, columns_, leading_dimension_);
  }

  ConstMatrixView<T> view() const noexcept {
    return ConstMatrixView<T>(
      data(), rows_, columns_, leading_dimension_
    );
  }

  T& operator()(std::size_t row, std::size_t column) {
    return view()(row, column);
  }

  const T& operator()(std::size_t row, std::size_t column) const {
    return view()(row, column);
  }

 private:
  std::vector<T, MatrixAllocator<T>> values_;
  std::size_t rows_ = 0;
  std::size_t columns_ = 0;
  std::size_t leading_dimension_ = 0;
};

template<class T>
ConstMatrixView<T> make_const_view(const T* data, std::size_t rows,
                                   std::size_t columns,
                                   std::size_t leading_dimension) noexcept {
  return ConstMatrixView<T>(data, rows, columns, leading_dimension);
}

template<class T>
MatrixView<T> make_view(T* data, std::size_t rows, std::size_t columns,
                        std::size_t leading_dimension) noexcept {
  return MatrixView<T>(data, rows, columns, leading_dimension);
}

}  // namespace core
}  // namespace fastpls

#endif
