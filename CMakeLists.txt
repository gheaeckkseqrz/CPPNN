cmake_minimum_required(VERSION 3.1)
project( ArtoNNLib VERSION 1.0 DESCRIPTION "Neural Network Runtime" )
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
find_package( OpenCL REQUIRED )
find_package( clBLAS REQUIRED )
include_directories( ${OpenCL_INCLUDE_DIRS} ${clBLAS_INCLUDE_DIR} )
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/OpenCLFuncs.cpp ${CMAKE_CURRENT_SOURCE_DIR}/OpenCLFuncs.h
  COMMAND source .env/bin/activate && python kernels.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Kernels/*.cl
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Kernels
  COMMENT "Generating code for OpenCLFuncs.")
add_library( ArtoNNLib
  Convolution.cpp
  MaxPooling.cpp
  Module.cpp
  OpenCL.cpp
  OpenCLFuncs.cpp
  ReflectionPadding.cpp
  Relu.cpp
  Sequential.cpp
  Tensor.cpp
  TorchConvolution.cpp
  TorchObject.cpp
  TorchLoader.cpp
  TorchMaxPooling.cpp
  TorchNumber.cpp
  TorchReflectionPadding.cpp
  TorchRelu.cpp
  TorchSequential.cpp
  TorchStorage.cpp
  TorchString.cpp
  TorchTable.cpp
  TorchTensor.cpp )
set_target_properties( ArtoNNLib PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties( ArtoNNLib PROPERTIES SOVERSION 1)
target_link_libraries( ArtoNNLib ${OpenCL_LIBRARY} ${CLBLAS_LIBRARIES} )

add_executable( TestNN Tests/main.cpp )
target_link_libraries( TestNN ArtoNNLib )

add_executable( runNN main.cpp )
target_link_libraries( runNN ArtoNNLib )
