// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Selected operations adapted from NVIDIA CUTLASS at
// 147295a3d4b75f3aeff247c25b8927cea9a7006a (see ../LICENSE.cutlass):
// cutlass/fast_math.h, cutlass/arch/{barrier.h,reg_reconfig.h}, and
// cute/arch/{cluster_sm90.hpp,copy_sm90.hpp,copy_sm100.hpp}.
// Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MSCCLPP_EXT_MEGAMOE_PRIMITIVES_CUH_
#define MSCCLPP_EXT_MEGAMOE_PRIMITIVES_CUH_

#include <cassert>
#include <cstdint>

#include "megamoe_specialization.hpp"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

// Only the positive-divisor/nonnegative-int task index path of CUTLASS FastDivmod is used.
struct TaskDivmod {
  int32_t divisor = 1;
  uint32_t multiplier = 0;
  uint32_t shiftRight = 0;

  constexpr TaskDivmod() = default;
  explicit constexpr TaskDivmod(int divisor_) : divisor(divisor_) {
    assert(divisor > 0);
    if (divisor == 1) return;
    unsigned p = 31;
    for (uint32_t v = uint32_t(divisor - 1); v != 0; v >>= 1) ++p;
    multiplier = uint32_t(((1ull << p) + uint32_t(divisor) - 1) / uint32_t(divisor));
    shiftRight = p - 32;
  }

#if defined(__CUDACC__)
  __host__ __device__ __forceinline__
#endif
      void
      operator()(int& quotient, int& remainder, int dividend) const {
#if defined(__CUDA_ARCH__)
    quotient = divisor != 1 ? int(__umulhi(uint32_t(dividend), multiplier) >> shiftRight) : dividend;
#else
    quotient = divisor != 1 ? int(((uint64_t(uint32_t(dividend)) * multiplier) >> 32) >> shiftRight) : dividend;
#endif
    remainder = dividend - quotient * divisor;
  }
};
static_assert(sizeof(TaskDivmod) == 12 && alignof(TaskDivmod) == 4);

// These are hardware IDs, not CUTLASS user IDs (which add eight).
constexpr uint32_t EpilogueBarrier = 1;
constexpr uint32_t TransformBarrier = 3;
constexpr uint32_t OutputOffloadSlotEmptyBarrier = 8;
constexpr uint32_t OutputOffloadSlotFullBarrier = 9;

#if defined(__CUDACC__)

// SM100f's full-warp election and CTA queries, from the pinned cluster_sm90.hpp.
__device__ __forceinline__ uint32_t electMoeLane() {
  uint32_t pred = 0, lane = 0;
  asm volatile("{\n\t.reg .b32 rx;\n\t.reg .pred px;\n\t"
               "elect.sync rx|px, %2;\n\t@px mov.s32 %1, 1;\n\tmov.s32 %0, rx;\n}"
               : "+r"(lane), "+r"(pred) : "r"(0xffffffffu));
  return pred;
}

__device__ __forceinline__ uint32_t moeClusterRank() {
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  return rank;
}

__device__ __forceinline__ dim3 moeClusterBlock() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(x));
  asm volatile("mov.u32 %0, %%cluster_ctaid.y;" : "=r"(y));
  asm volatile("mov.u32 %0, %%cluster_ctaid.z;" : "=r"(z));
  return {x, y, z};
}

__host__ __device__ constexpr uint32_t accumulatorStage(uint32_t base, int stage) {
  return base + 128 * stage;
}

__device__ __forceinline__ void namedBarrierSync(uint32_t threads, uint32_t id) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

__device__ __forceinline__ void namedBarrierArrive(uint32_t threads, uint32_t id) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

// Selected CUTLASS arch/barrier.h::arrive_and_expect_tx at 147295a3.
// The caller elects the leader and chooses the transaction byte count.
__device__ __forceinline__ void arriveMoeTmaExpect(const uint64_t* bar, uint32_t bytes) {
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{\n\tmbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;\n}"
               :: "r"(bytes), "r"(address) : "memory");
}

__device__ __forceinline__ void fenceBarrierInit() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void fenceAsyncSharedView() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void clusterSync() {
  asm volatile("barrier.cluster.arrive.aligned;" ::);
  asm volatile("barrier.cluster.wait.aligned;" ::);
}

// The MegaMoE translation units require the SM100-family device view. Keep the
// upstream explicit reconfiguration override when the compiler is given one.
template <int Registers>
__device__ __forceinline__ void warpgroupRegAlloc() {
#if !defined(CUDA_CTA_RECONFIG_ACTIVATED) || CUDA_CTA_RECONFIG_ACTIVATED
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(Registers));
#endif
}

template <int Registers>
__device__ __forceinline__ void warpgroupRegDealloc() {
#if !defined(CUDA_CTA_RECONFIG_ACTIVATED) || CUDA_CTA_RECONFIG_ACTIVATED
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(Registers));
#endif
}

// Selected tcgen05 allocation/release from CUTLASS 147295a3,
// cute/arch/tmem_allocator_sm100.hpp. The warp/CTA participation stays at call sites.
template <int ClusterSize>
struct MegaMoeTmemAllocator {
  static_assert(ClusterSize == 1 || ClusterSize == 2);

  __device__ __forceinline__ void allocate(int columns, uint32_t* dst) {
    uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
#if defined(CUTE_ARCH_TCGEN05_TMEM_EXCLUSIVE_ENABLED)
    asm volatile("tcgen05.alloc.exclusive.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(address), "r"(columns), "n"(ClusterSize));
#else
    asm volatile("tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(address), "r"(columns), "n"(ClusterSize));
#endif
  }

  __device__ __forceinline__ void free(uint32_t tmem, int columns) {
#if defined(CUTE_ARCH_TCGEN05_TMEM_EXCLUSIVE_ENABLED)
    asm volatile("{\n\ttcgen05.dealloc.exclusive.cta_group::%2.sync.aligned.b32 %0, %1;\n}"
                 :: "r"(tmem), "r"(columns), "n"(ClusterSize));
#else
    asm volatile("{\n\ttcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;\n}"
                 :: "r"(tmem), "r"(columns), "n"(ClusterSize));
#endif
  }

  __device__ __forceinline__ void release_allocation_lock() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::%0.sync.aligned;" :: "n"(ClusterSize));
  }
};

// The selected producer state starts empty/opposite phase. The pipe still
// owns wraparound, wait/release ordering and all consumer state.
template <class Pipeline>
__device__ __forceinline__ typename Pipeline::PipelineState producerStart() {
  return {0, 1u, 0u};
}

__device__ __forceinline__ void waitTmemLoad() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }

__device__ __forceinline__ void waitTmemStore() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

// For each eight-column group, thread t gets lanes t/4 and t/4+8,
// at columns 2*(t%4) and 2*(t%4)+1: the transposed stmatrix operand order.
template <int Groups>
__device__ __forceinline__ void loadFc2Accumulators(uint32_t address, uint32_t (&data)[4 * Groups]) {
  if constexpr (Groups == 4) {
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        : "=r"(data[0]), "=r"(data[1]), "=r"(data[2]), "=r"(data[3]), "=r"(data[4]), "=r"(data[5]), "=r"(data[6]),
          "=r"(data[7]), "=r"(data[8]), "=r"(data[9]), "=r"(data[10]), "=r"(data[11]), "=r"(data[12]), "=r"(data[13]),
          "=r"(data[14]), "=r"(data[15])
        : "r"(address));
  } else if constexpr (Groups == 2) {
    asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(data[0]), "=r"(data[1]), "=r"(data[2]), "=r"(data[3]), "=r"(data[4]), "=r"(data[5]),
                   "=r"(data[6]), "=r"(data[7])
                 : "r"(address));
  } else {
    static_assert(Groups == 1);
    asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(data[0]), "=r"(data[1]), "=r"(data[2]), "=r"(data[3])
                 : "r"(address));
  }
}

__device__ __forceinline__ void storeTransposedB16x4(void* destination, const uint32_t (&words)[4]) {
  const auto address = uint32_t(__cvta_generic_to_shared(destination));
  asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1,%2,%3,%4};" ::"r"(address), "r"(words[0]),
               "r"(words[1]), "r"(words[2]), "r"(words[3]));
}

#endif  // defined(__CUDACC__)

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_PRIMITIVES_CUH_
