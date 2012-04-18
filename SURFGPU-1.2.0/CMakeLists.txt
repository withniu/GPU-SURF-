cmake_minimum_required(VERSION 2.6)

PROJECT (SURFGPU)

INCLUDE(${CMAKE_SOURCE_DIR}/CMake/cuda/FindCUDA.cmake)
include(${CMAKE_SOURCE_DIR}/CMake/FindCUDPP.cmake)
find_package(OpenCV REQUIRED)

#set(CUDA_BUILD_CUBIN ON)
set(CUDA_BUILD_EMULATION OFF)
set(CUDA_NVCC_FLAGS "-arch=sm_11")

option(CUDA_24BIT_IMUL "Use 24bit integer multiplication for IMUL macros" ON)
if(CUDA_24BIT_IMUL)
	add_definitions(-DCUDA_24BIT_IMUL)
endif()

option(SURF_MATCH_SIMPLE "Use simple SURF descriptor matching" OFF)
if(SURF_MATCH_SIMPLE)
	add_definitions(-DSURF_MATCH_SIMPLE)
endif()

# Needed for M_PI and friends on Windows
if(WIN32)
	add_definitions(-D_USE_MATH_DEFINES)
endif()

INCLUDE_DIRECTORIES(
${CMAKE_SOURCE_DIR}
${CUDA_TOOLKIT_INCLUDE}
${OpenCV_INCLUDE_DIR}
${CUDA_INCLUDE_DIRS}
${CUDPP_INCLUDE_DIR}
)

CUDA_INCLUDE_DIRECTORIES(
${CMAKE_SOURCE_DIR}
${CUDA_TOOLKIT_INCLUDE}
${OpenCV_INCLUDE_DIR}
${CUDA_INCLUDE_DIRS}
)

LINK_DIRECTORIES(
	${CUDA_SDK_LIB_DIR}
)

set_source_files_properties(cuda/buildSURFDescriptorsGPU.cu PROPERTIES
	COMPILE_FLAGS "--maxrregcount 18")
CUDA_ADD_EXECUTABLE(surf
	fasthessian.cpp
	integral.cpp
	ipoint.cpp
#	main.cpp
	main_kinect.cpp
	surf.cpp
	utils.cpp
	cuda/buildSURFDescriptorsGPU.cu
	cuda/convertRGB2GrayFloatGPU.cu
	cuda/convertRGBA2GrayFloatGPU.cu
	cuda/detectIpointOrientationsGPU.cu
	cuda/getMatchesGPU.cpp
	cuda/helper_funcs.cpp
	cuda/buildDetGPU.cu
	cuda/nonMaxGPU.cu
	cuda/matchSURFKeypointsGPU.cu
	cuda/normalizeSURFDescriptorsGPU.cu
	cuda/strerror_cudpp.cpp
	cuda/transposeGPU.cu
)

TARGET_LINK_LIBRARIES(surf
${CUDA_LIBRARIES}
${OpenCV_LIBS}
${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDPP_LIBRARY}
)

