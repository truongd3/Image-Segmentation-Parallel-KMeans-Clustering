cmake_minimum_required(VERSION 3.27)

project(image_segmentation          
        VERSION 1.0
        LANGUAGES CXX CUDA)         

# ────────────────────────────── Global settings ──────────────────────────────
# Make CMake put -I flags on the command-line instead of in a response file
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 50)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)          
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)

find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)


include_directories(SYSTEM
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# ────────────────────────────── Build switches ───────────────────────────────
option(BUILD_SERIAL    "CPU-only implementation"           OFF)
option(BUILD_CUDA      "Pure-CUDA implementation"          ON)
option(BUILD_MPI       "MPI implementation"                OFF)
option(BUILD_MPI_CUDA  "Hybrid MPI + CUDA implementation"  OFF)


# ────────────────────────────── Common helper library ────────────────────────
add_library(common
    src/common/image_utils.cpp
    src/common/kmeans_utils.cpp
)
target_include_directories(common
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

# ────────────────────────────── Serial (CPU-only) build ──────────────────────
if(BUILD_SERIAL)
    add_executable(img_seg_serial
        src/serial/main_serial.cpp
        src/serial/img_seg_serial.cpp
        src/serial/kmeans.cpp
    )
    target_link_libraries(img_seg_serial PRIVATE common ${OpenCV_LIBS})
endif()

# ────────────────────────────── CUDA-only build ──────────────────────────────

if(BUILD_CUDA)
    enable_language(CUDA)   
    add_executable(img_seg_gpu
        src/gpu/main_gpu.cpp
        src/gpu/img_seg_gpu.cu
    )
    target_link_libraries(img_seg_gpu PRIVATE common ${OpenCV_LIBS})
    set_target_properties(img_seg_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE  ON
    )
endif()

# ────────────────────────────── MPI build ────────────────────────────────────
if(BUILD_MPI)
    find_package(MPI REQUIRED COMPONENTS C CXX)
    add_executable(img_seg_mpi
        src/mpi/main_mpi.cpp     
        src/mpi/img_seg_mpi.cpp
        src/mpi/kmeans.cpp
    )
    target_link_libraries(img_seg_mpi PRIVATE MPI::MPI_CXX common ${OpenCV_LIBS})
endif()

# ────────────────────────────── MPI + CUDA build ─────────────────────────────
if(BUILD_MPI_CUDA)
    enable_language(CUDA)
    find_package(MPI REQUIRED COMPONENTS C CXX)
    add_executable(img_seg_mpi_cuda
        src/mpi_cuda/main_mpi_cuda.cu   # placeholder
    )
    target_link_libraries(img_seg_mpi_cuda PRIVATE MPI::MPI_CXX common)
    set_target_properties(img_seg_mpi_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

endif()

