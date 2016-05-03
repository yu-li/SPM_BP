cmake_minimum_required(VERSION 2.8)
project( spmbpOF )

set (CMAKE_C_FLAGS_RELEASE "-O3")
# set (CMAKE_BUILD_TYPE RelWithDebInfo)
set (CMAKE_BUILD_TYPE Release)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP Found")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package (OpenCV REQUIRED)
if (UNIX)
include_directories (SYSTEM ${OPENCV_INCLUDE_DIRS})
endif(UNIX)
if(MSVC)
include_directories(C:\\opencv\\build\\include)
link_directories(C:\\opencv\\build\\lib)
endif(MSVC)
include_directories (code)

file(GLOB FILTER_FILE 
    code/FGS*
    code/CF*)

file(GLOB SUPERPIXEL_FILE 
    code/SLIC* 
    code/Superpixels*)

file(GLOB FLOW_MANIPULATION_FILE 
    code/colorcode* 
    code/FlowInputOutput*)

file(GLOB COMMON_FILE 
    code/Common_Datastructure_*
    )

file(GLOB OF_FILE 
    code/opticalFlow*
    )

file(GLOB SPMBP_FILE 
    code/spmbp.h
    code/spmbp.cpp
    )

add_executable(spmbpOF 
    code/spmbpOF.cpp
    ${FILTER_FILE}
    ${SUPERPIXEL_FILE}
    ${FLOW_MANIPULATION_FILE}
    ${COMMON_FILE}
    ${OF_FILE}
    ${SPMBP_FILE}
    )

source_group("filter" FILES ${FILTER_FILE})
source_group("super pixel" FILES ${SUPERPIXEL_FILE})
source_group("flow_io" FILES ${FLOW_MANIPULATION_FILE})
source_group("util" FILES ${COMMON_FILE})
source_group("optical flow" FILES ${OF_FILE})
source_group("spmbp" FILES ${SPMBP_FILE})

target_link_libraries(spmbpOF ${OpenCV_LIBS})

