cmake_minimum_required(VERSION 2.8)
project( spmbpOF )

set (CMAKE_C_FLAGS_RELEASE "-O3")
set (CMAKE_BUILD_TYPE Release)

# if(MSVC)
# set(OPENCV_INCLUDE C:\\opencv\\build\\include)
# set(OPENCV_LIB_PATH C:\\opencv\\build\\x64\\vc11\\lib)
# include_directories(
#     ${OPENCV_INCLUDE}
# )
#
# link_directories(
#     ${OPENCV_LIB_PATH}
# )
# endif(MSVC)

# if (UNIX)
find_package (OpenCV REQUIRED)
include_directories (SYSTEM ${OPENCV_INCLUDE_DIRS})
include_directories (code)
# endif(UNIX)

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

# if (MSVC)
# target_link_libraries(spmbpOF 
#     debug opencv_ts300d.lib optimized opencv_ts300.lib
#     debug opencv_world300d.lib  optimized opencv_world300.lib
# )
# endif(MSVC)

# if (UNIX)
target_link_libraries(spmbpOF ${OpenCV_LIBS})
# endif(UNIX)


