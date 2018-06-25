cmake_minimum_required(VERSION 3.1)
project( ArtoNNLib VERSION 1.0 DESCRIPTION "Neural Network Runtime" )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
find_package( OpenCL REQUIRED )
find_package( clBLAS REQUIRED )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCL_INCLUDE_DIRS} ${clBLAS_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/library ${CMAKE_CURRENT_SOURCE_DIR}/external)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/library/OpenCLFuncs.cpp ${CMAKE_CURRENT_SOURCE_DIR}/library/OpenCLFuncs.h
  COMMAND source .env/bin/activate && python kernels.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/library/Kernels/*.cl
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/library/Kernels
  COMMENT "Generating code for OpenCLFuncs.")
add_library( ArtoNNLib
  library/ComputeGraph.cpp
  library/Convolution.cpp
  library/Kmeans.cpp
  library/MaxPooling.cpp
  library/Module.cpp
  library/Node.cpp
  library/NodeAdd.cpp
  library/NodeMul.cpp
  library/OpenCL.cpp
  library/OpenCLFuncs.cpp
  library/ParametricModel.cpp
  library/PCA.cpp
  library/PixelShuffle.cpp
  library/Placeholder.cpp
  library/ReflectionPadding.cpp
  library/Relu.cpp
  library/Segmenter.cpp
  library/Sequential.cpp
  library/SingularValueDecomposition.cpp
  library/SpatialUpsamplingNearest.cpp
  library/Tanh.cpp
  library/Tensor.cpp
  library/TensorflowConvolution.cpp
  library/TensorflowFactory.cpp
  library/TensorflowParser.cpp
  library/TensorflowTensor.cpp
  library/TextureLibrary.cpp
  library/TorchConvolution.cpp
  library/TorchObject.cpp
  library/TorchLoader.cpp
  library/TorchMaxPooling.cpp
  library/TorchNumber.cpp
  library/TorchReflectionPadding.cpp
  library/TorchRelu.cpp
  library/TorchSequential.cpp
  library/TorchSpatialUpsamplingNearest.cpp
  library/TorchStorage.cpp
  library/TorchString.cpp
  library/TorchTable.cpp
  library/TorchTensor.cpp
  library/VGGPreprocessing.cpp
  library/WCT.cpp )

set_target_properties( ArtoNNLib PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties( ArtoNNLib PROPERTIES SOVERSION 1)
target_link_libraries( ArtoNNLib ${OpenCL_LIBRARY} ${CLBLAS_LIBRARIES} ${OpenCV_LIBS})# -lstdc++fs)

add_executable( TestNN tests/main.cpp )
target_link_libraries( TestNN ArtoNNLib )

add_executable( Segmenter samples/Segmenter.cpp MatToTensor.cpp )
target_link_libraries( Segmenter ArtoNNLib ${OpenCV_LIBS} )

add_executable( Upres samples/Upres.cpp MatToTensor.cpp )
target_link_libraries( Upres ArtoNNLib ${OpenCV_LIBS} )

add_executable( PCABenchmark samples/PCABenchmark.cpp )
target_link_libraries( PCABenchmark ArtoNNLib )

add_executable( ConvBenchmark samples/ConvBenchmark.cpp )
target_link_libraries( ConvBenchmark ArtoNNLib )

add_executable( TextureLibrary samples/TextureLibrary.cpp MatToTensor.cpp )
target_link_libraries( TextureLibrary ArtoNNLib ${OpenCV_LIBS} )

add_executable( StyleTransfer samples/StyleTransfer.cpp MatToTensor.cpp )
target_link_libraries( StyleTransfer ArtoNNLib ${OpenCV_LIBS} )

add_executable( VideoStyleTransfer samples/VideoStyleTransfer.cpp MatToTensor.cpp )
target_link_libraries( VideoStyleTransfer ArtoNNLib ${OpenCV_LIBS} )
