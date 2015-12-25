project(pico-cascade)
cmake_minimum_required(VERSION 3.0)

#include(ExternalProject)
#include(CheckIncludeFileCXX)

#find_package(OpenCV REQUIRED)
#find_package(Boost COMPONENTS filesystem system QUIET)
#if (Boost_FOUND)
#	add_definitions(-DHAVE_BOOST)
#endif()
find_package(CUDA QUIET)
if (CUDA_FOUND)
	add_definitions(-DHAVE_CUDA)
	include_directories(${CUDA_INCLUDE_DIRS})
endif()


# some folder as a string literal
#add_definitions(-DDEBUG_MODEL_FOLDER="${CMAKE_SOURCE_DIR}/../data/models")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

#include_directories(subproject/include)
#add_subdirectory(subproject)

# some local project
#set(CLANDMARK_DIR ${CMAKE_SOURCE_DIR}/../clandmark/libclandmark)
#ExternalProject_Add(
#	libclandmark
#	PREFIX clandmark
#	DOWNLOAD_COMMAND ""
#	INSTALL_COMMAND ""
#	SOURCE_DIR ${CLANDMARK_DIR})
#ExternalProject_Get_Property(libclandmark BINARY_DIR)
#link_directories(${BINARY_DIR})
#include_directories(${CLANDMARK_DIR}/..)

# some project from GIT
#ExternalProject_Add(
#	caffe
#	PREFIX caffe
#	GIT_REPOSITORY git@github.com:BVLC/caffe.git
#	GIT_TAG master)

# setup some project
#find_package(Caffe)
#add_definitions(${Caffe_DEFINITIONS})
#include_directories(${Caffe_INCLUDE_DIRS})
#ExternalProject_Get_property(caffe binary_dir)
#link_directories(${binary_dir}/lib)
#add_definitions(-DHAVE_CAFFE)

# sources
set(RUNTIME_SRC
	rnt/picornt.cpp
	rnt/picornt.h
)

set(GEN_SRC
	gen/picogen.cpp
)

set(TRAINER_SRC
	gen/picolrn.cpp
)

set(CUPICO_SRC
	rnt/detect-cuda.cu
	rnt/detect-cuda.h
)

# check include file presence
#check_include_file_cxx(opencv2/face/facerec.hpp HAVE_OPENCV_CONTRIB)
#if(HAVE_OPENCV_CONTRIB)
#	add_definitions(-DHAVE_OPENCV_CONTRIB)
#endif(HAVE_OPENCV_CONTRIB)

set(CMAKE_VERBOSE_MAKEFILE ON)

# common libraries
set(COMMON_LIBRARIES
	${OpenCV_LIBS}
	${Boost_LIBRARIES}
	${Caffe_LIBRARIES}
	${PROTOBUF_LIBRARIES})

add_executable(picogen gen/picogen.cpp)
#add_dependencies(my-lib subproject)
#target_link_libraries(my-lib ${COMMON_LIBRARIES})

add_executable(picolrn gen/picolrn.cpp)
set_target_properties(picolrn PROPERTIES LINK_FLAGS "-fopenmp")
#add_dependencies(my-lib subproject)
#target_link_libraries(picolrn
#	rt)

if (CUDA_FOUND)
	cuda_compile(CUDA_OBJ ${CUPICO_SRC})
	add_library(pico ${RUNTIME_SRC} ${CUDA_OBJ})
endif()
