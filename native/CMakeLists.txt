cmake_minimum_required(VERSION 3.27)

project(mlx_cce_runtime_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(
  Python 3.8
  COMPONENTS Interpreter Development.Module
  REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)
execute_process(
  COMMAND
  "${Python_EXECUTABLE}"
  -c
  "import pathlib, mlx.core; print(pathlib.Path(mlx.core.__file__).resolve().parents[2])"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_SOURCE_ROOT)

add_library(mlx_cce_runtime_native)
target_sources(
  mlx_cce_runtime_native
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/dense_cce_backward.cpp)
target_include_directories(
  mlx_cce_runtime_native
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}
  PRIVATE ${MLX_SOURCE_ROOT} ${MLX_SOURCE_ROOT}/mlx ${MLX_INCLUDE_DIRS})
target_link_libraries(mlx_cce_runtime_native PUBLIC mlx)

if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET
    mlx_cce_runtime_native_metallib
    TITLE
    mlx_cce_runtime_native
    SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/dense_cce_backward.metal
    ${MLX_SOURCE_ROOT}/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal
    INCLUDE_DIRS
    ${CMAKE_CURRENT_LIST_DIR}
    ${MLX_SOURCE_ROOT}
    ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  add_dependencies(mlx_cce_runtime_native mlx_cce_runtime_native_metallib)
endif()

nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE mlx_cce_runtime_native)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
