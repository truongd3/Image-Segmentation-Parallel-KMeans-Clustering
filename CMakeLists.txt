cmake_minimum_required(VERSION 3.30)
project(image-segmentation VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS} 
    ${PROJECT_SOURCE_DIR}/include
)

# Version 1: CPU-only executable
add_executable(seg_cpu src/serial/main_serial.cpp src/common/image_utils.cpp)

target_link_libraries( seg_cpu ${OpenCV_LIBS} )

