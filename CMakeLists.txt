cmake_minimum_required(VERSION 3.20)
project(fastPLS_py VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core MODULE
  src/bindings.cpp
  src/native_backend.cpp
)
target_include_directories(_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/fastpls/include
)

if(APPLE)
  target_compile_definitions(_core PRIVATE FASTPLS_PY_ACCELERATE=1)
  target_link_libraries(_core PRIVATE "-framework Accelerate")
else()
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  target_compile_definitions(_core PRIVATE FASTPLS_PY_SYSTEM_BLAS=1)
  target_link_libraries(_core PRIVATE BLAS::BLAS LAPACK::LAPACK)
endif()

if(MSVC)
  target_compile_options(_core PRIVATE /O2 /EHsc)
else()
  target_compile_options(_core PRIVATE -O3)
endif()

install(TARGETS _core LIBRARY DESTINATION fastpls)

