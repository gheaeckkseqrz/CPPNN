cmake_minimum_required(VERSION 3.1)
project( ArtoNN )
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
find_package( OpenCL REQUIRED )
include_directories( ${OpenCL_INCLUDE_DIRS} )
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/OpenCLFuncs.cpp ${CMAKE_CURRENT_SOURCE_DIR}/OpenCLFuncs.h
  COMMAND source .env/bin/activate && python kernels.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Kernels/*.cl
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Kernels
  COMMENT "Generating code for OpenCLFuncs.")
add_executable( TestNN main.cpp
  Convolution.cpp
  MaxPooling.cpp
  Module.cpp
  OpenCL.cpp
  OpenCLFuncs.cpp
  Relu.cpp
  Sequential.cpp
  Tensor.cpp
  TorchConvolution.cpp
  TorchObject.cpp
  TorchLoader.cpp
  TorchMaxPooling.cpp
  TorchNumber.cpp
  TorchRelu.cpp
  TorchSequential.cpp
  TorchStorage.cpp
  TorchString.cpp
  TorchTable.cpp
  TorchTensor.cpp )
target_link_libraries( TestNN ${OpenCL_LIBRARY} )
