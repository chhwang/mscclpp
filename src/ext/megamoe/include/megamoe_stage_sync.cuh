// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Selected mbarrier/UMMA leaves and ring state from NVIDIA CUTLASS at
// 147295a3d4b75f3aeff247c25b8927cea9a7006a, cutlass/arch/barrier.h and
// cutlass/pipeline/sm90_pipeline.hpp. See ../LICENSE.cutlass.
// Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MSCCLPP_EXT_MEGAMOE_STAGE_SYNC_CUH_
#define MSCCLPP_EXT_MEGAMOE_STAGE_SYNC_CUH_

#include <cstdint>
#include <mscclpp/device.hpp>

#include "megamoe_kernel.hpp"
#include "megamoe_primitives.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

// The consumer starts at (0,0,0); producerStart starts in the opposite phase.
template <int Slots>
struct MoeStageState {
  static_assert(Slots > 0);
  int stage = 0;
  uint32_t parity = 0;
  uint32_t completed = 0;

  __host__ __device__ constexpr int index() const { return stage; }
  __host__ __device__ constexpr uint32_t phase() const { return parity; }
  __host__ __device__ constexpr void operator++() {
    ++stage;
    ++completed;
    if (stage == Slots) {
      stage = 0;
      parity ^= 1;
    }
  }
};

template <int Slots>
struct MoeStageStorage {
  uint64_t full_barrier_[Slots];
  uint64_t empty_barrier_[Slots];
};

enum class MoeStageRole { NonParticipant, Producer, Consumer };
enum class MoeStageReady : uint32_t { WaitAgain = 0, WaitDone = 1 };

MSCCLPP_DEVICE_INLINE int stageWarp() {
  return __shfl_sync(0xffffffff, threadIdx.x / WARP_SIZE, 0);
}

MSCCLPP_DEVICE_INLINE void stageCheckRole(MoeStageRole actual, MoeStageRole expected) {
#ifndef NDEBUG
  if (actual != expected) asm volatile("brkpt;" ::);
#else
  (void)actual;
  (void)expected;
#endif
}

MSCCLPP_DEVICE_INLINE void stageInit(uint64_t* bar, uint32_t arrivals) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{\n\tmbarrier.init.shared::cta.b64 [%1], %0;\n}"
               :: "r"(arrivals), "r"(addr) : "memory");
}

template <int Slots>
MSCCLPP_DEVICE_INLINE void stageInitPair(MoeStageStorage<Slots>& storage, uint32_t producer, uint32_t consumer) {
  if (electMoeLane()) {
#pragma unroll
    for (int i = 0; i < Slots; ++i) {
      stageInit(&storage.full_barrier_[i], producer);
      stageInit(&storage.empty_barrier_[i], consumer);
    }
  }
}

MSCCLPP_DEVICE_INLINE void stageWait(const uint64_t* bar, uint32_t phase) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t ticks = 0x989680;
  asm volatile("{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n\t"
               "@P1 bra DONE;\n\tbra LAB_WAIT;\n\tDONE:\n}"
               :: "r"(addr), "r"(phase), "r"(ticks) : "memory");
}

MSCCLPP_DEVICE_INLINE bool stageTryWait(const uint64_t* bar, uint32_t phase) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t ready;
  asm volatile("{\n\t.reg .pred P1;\n\tmbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
               "selp.b32 %0, 1, 0, P1;\n}"
               : "=r"(ready) : "r"(addr), "r"(phase) : "memory");
  return static_cast<bool>(ready);
}

MSCCLPP_DEVICE_INLINE bool stageTestWait(const uint64_t* bar, uint32_t phase) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t ready;
  asm volatile("{\n\t.reg .pred P1;\n\t.reg .pred P2;\n\tsetp.eq.u32 P2, %3, 1;\n\t"
               "@P2 mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
               "selp.b32 %0, 1, 0, P1;\n}"
               : "=r"(ready) : "r"(addr), "r"(phase), "r"(1u) : "memory");
  return static_cast<bool>(ready);
}

MSCCLPP_DEVICE_INLINE void stageArriveCluster(const uint64_t* bar, uint32_t target, uint32_t pred) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  if (pred) {
    asm volatile("{\n\t.reg .b32 remAddr32;\n\tmapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                 "mbarrier.arrive.shared::cluster.b64 _, [remAddr32];\n}"
                 :: "r"(addr), "r"(target) : "memory");
  }
}

MSCCLPP_DEVICE_INLINE void stageArriveLeader(const uint64_t* bar) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)) & 0xfeffffffu;
  asm volatile("{\n\tmbarrier.arrive.shared::cluster.b64 _, [%0];\n}" :: "r"(addr) : "memory");
}

template <int ClusterSize>
MSCCLPP_DEVICE_INLINE void stageCommit(const uint64_t* bar, uint16_t mask) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  if (electMoeLane()) {
    if constexpr (ClusterSize == 2) {
      asm volatile("{\n\t"
                   "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n}"
                   :: "r"(addr), "h"(mask) : "memory");
    } else {
      static_assert(ClusterSize == 1);
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                   :: "r"(addr) : "memory");
    }
  }
}

// An adjacent MMA two-CTA group; only x varies in the admitted cluster.
__host__ __device__ constexpr uint16_t stagePairMask(uint32_t x) {
  const auto first = x / 2 * 2;
  return uint16_t((1u << first) | (1u << (first + 1)));
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_STAGE_SYNC_CUH_
