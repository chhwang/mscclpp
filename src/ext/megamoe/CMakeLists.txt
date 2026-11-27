# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

if(NOT MSCCLPP_USE_CUDA OR MSCCLPP_USE_ROCM)
    message(FATAL_ERROR "Native MegaMoE requires NVIDIA CUDA; ROCm is not supported.")
endif()
if(CUDAToolkit_VERSION VERSION_LESS "13.0")
    message(FATAL_ERROR "Native MegaMoE requires CUDA 13.0 or newer.")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.0")
    message(FATAL_ERROR "Native MegaMoE requires CUDA nvcc/ptxas >=13.0.")
endif()

add_library(mscclpp_megamoe SHARED runtime.cc megamoe_w8a16.cu megamoe_w4a8.cu megamoe_launch.cu)
target_include_directories(mscclpp_megamoe PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)
target_compile_options(mscclpp_megamoe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda>
)
target_link_libraries(mscclpp_megamoe PUBLIC mscclpp PRIVATE ${GPU_LIBRARIES} ${CMAKE_DL_LIBS})
target_compile_definitions(mscclpp_megamoe PRIVATE MSCCLPP_USE_CUDA)
set_target_properties(mscclpp_megamoe PROPERTIES
    CUDA_ARCHITECTURES "100f-real"
    POSITION_INDEPENDENT_CODE ON
    VERSION ${MSCCLPP_VERSION}
    SOVERSION ${MSCCLPP_SOVERSION}
    INSTALL_RPATH "$ORIGIN"
)
install(TARGETS mscclpp_megamoe LIBRARY DESTINATION ${INSTALL_PREFIX}/lib)
install(FILES LICENSE.cutlass DESTINATION ${INSTALL_PREFIX}/licenses/mscclpp_megamoe)

set(MEGAMOE_JIT_INSTALL_DIR "${INSTALL_PREFIX}/share/mscclpp/megamoe")
install(FILES megamoe_w8a16.cu megamoe_w4a8.cu megamoe_launch.cu megamoe_jit.cu LICENSE.cutlass "${PROJECT_SOURCE_DIR}/LICENSE"
    DESTINATION "${MEGAMOE_JIT_INSTALL_DIR}")
install(FILES
    include/megamoe_accumulator_pipe.cuh
    include/megamoe_activation_tma.hpp
    include/megamoe_collective.cuh
    include/megamoe_combine.cuh
    include/megamoe_load_pipes.cuh
    include/megamoe_stage_sync.cuh
    include/megamoe_tma_issue.cuh
    include/megamoe_dispatch.cuh
    include/megamoe_math.cuh
    include/megamoe_primitives.cuh
    include/megamoe_quantization.cuh
    include/megamoe_w4a8_collective.cuh
    include/megamoe_w4a8_roles.cuh
    include/megamoe_w8a16_collective.cuh
    include/megamoe_trace.cuh
    include/megamoe_device.cuh
    include/megamoe_epilogue.cuh
    include/megamoe_mma.cuh
    include/megamoe_w8a16_roles.cuh
    include/megamoe_routing.cuh
    include/megamoe_tasks.cuh
    include/megamoe_kernel.hpp
    include/megamoe_jit.hpp
    include/megamoe_specialization.hpp
    DESTINATION "${MEGAMOE_JIT_INSTALL_DIR}/include")
