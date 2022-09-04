cmake_minimum_required(VERSION 3.10.2)
project(ql)
add_compile_options(
        -Werror
        -Wall
        -lpthread
)
set(CMAKE_CXX_STANDARD 14)
if(NOT CMAKE_CXX_EXTENSIONS)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()
find_package(OpenCV REQUIRED)
include_directories(/usr/local/include/opencv4)
add_executable(ql main.cpp)
target_link_libraries(ql
        /usr/local/lib/libopencv_core.so
        /usr/local/lib/libopencv_imgcodecs.so
        /usr/local/lib/libopencv_imgproc.so.4.6.0
        /usr/local/lib/libopencv_highgui.so.4.6.0
        /lib/x86_64-linux-gnu/libpthread.so.0)
link_libraries(/usr/local/lib)

