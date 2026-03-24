cmake_minimum_required(VERSION 3.18)
project(tokffi LANGUAGES CXX)

find_package(tvm_ffi CONFIG REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(3rdparty/tokenizers-cpp)

add_library(tokffi SHARED
  src/tokffi.cc
)

target_compile_features(tokffi PRIVATE cxx_std_17)
target_include_directories(tokffi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tokffi PRIVATE tokenizers_cpp)

# 推奨: TVM-FFI 側の helper に include / shared lib / debug symbol 設定を任せる
tvm_ffi_configure_target(tokffi
  LINK_SHARED ON
  LINK_HEADER ON
  DEBUG_SYMBOL OFF
)

# helper が使えない場合の代替:
# target_link_libraries(tokffi PRIVATE tvm_ffi::shared tvm_ffi::header)