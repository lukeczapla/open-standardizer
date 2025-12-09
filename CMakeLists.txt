cmake_minimum_required(VERSION 3.20)

option(OPEN_STANDARDIZER_ENABLE_CUDA "Build open-standardizer with C++/CUDA support" OFF)

if(OPEN_STANDARDIZER_ENABLE_CUDA)
    message(STATUS "open-standardizer: CUDA support ENABLED")
    enable_language(CXX)
    enable_language(CUDA)


    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

    # Make sure CUDA knows what to target (A100 sm_80 + L4 sm_89 by default)
    # Put this in the CACHE so CMake's internal try-compile targets see it.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "CUDA architectures" FORCE)
    endif()

    # Point CMake at custom find module
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

    find_package(pybind11 CONFIG REQUIRED)
    find_package(RDKit REQUIRED)          # resolved via cmake/FindRDKit.cmake
    find_package(CUDAToolkit REQUIRED)    # CUDA::cudart

    # Python extension name must match PYBIND11_MODULE(standardize_gpu, m)
    # and the Python side import: `import standardize_gpu`
    pybind11_add_module(standardize_gpu
        src/gpu/standardize_bindings.cpp
        src/gpu/kernels.cu
        src/gpu/stereo.cu
        src/gpu/aromaticity.cu
        src/gpu/mesomerizer.cu
        src/gpu/bond_infer.cu
        src/gpu/charge_normalize.cu
        src/gpu/core_ops.cu
        src/gpu/tautomerize.cu
        src/gpu/standardize_cuda.cu
        src/gpu/core_ops_batch.cpp
    )

    target_link_libraries(standardize_gpu
        PRIVATE
            RDKit::RDKit
            CUDA::cudart
    )

    target_include_directories(standardize_gpu
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu
            ${CMAKE_CURRENT_SOURCE_DIR}/gpu
    )

    set_target_properties(standardize_gpu PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

else()
    message(STATUS "open-standardizer: CUDA support DISABLED (CPU-only build)")
endif()
