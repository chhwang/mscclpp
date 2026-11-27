// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Selected protocols from CUTLASS 147295a3d4b75f3aeff247c25b8927cea9a7006a
// (cutlass/pipeline/{sm100_pipeline,sm90_pipeline}.hpp). See ../LICENSE.cutlass.
// Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MSCCLPP_EXT_MEGAMOE_LOAD_PIPES_CUH_
#define MSCCLPP_EXT_MEGAMOE_LOAD_PIPES_CUH_

#include "megamoe_stage_sync.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

// W4 A/B and W8 BF16 B: CTA0's issuing lane publishes the exact byte count
// after acquire; TMA completes FULL. MMA completion releases EMPTY.
template <int Slots, int ClusterSize>
struct MoeTmaUmmaPipe {
  static_assert(ClusterSize == 1 || ClusterSize == 2);
  static constexpr int Stages = Slots;
  using SharedStorage = MoeStageStorage<Slots>;
  using PipelineState = MoeStageState<Slots>;
  using Role = MoeStageRole;

  MSCCLPP_DEVICE_INLINE MoeTmaUmmaPipe(SharedStorage& storage, Role role)
      : full_(storage.full_barrier_), empty_(storage.empty_barrier_), role_(role) {}

  __host__ __device__ static constexpr uint32_t producerArrivals(bool sharedW4) { return sharedW4 ? 2 : 1; }

  MSCCLPP_DEVICE_INLINE static void init(SharedStorage& storage, int initializingWarp) {
    if (stageWarp() == initializingWarp) stageInitPair(storage, producerArrivals(false), 1);
    fenceBarrierInit();
  }

  MSCCLPP_DEVICE_INLINE static void initSharedW4(SharedStorage& storage, int initializingWarp, int warp) {
    // This path defers the fence to the kernel's existing collective init
    // boundary. Two producers publish separately to the same FULL phase.
    if (warp == initializingWarp) stageInitPair(storage, producerArrivals(true), 1);
  }

  __host__ __device__ static constexpr uint16_t w4Mask(uint32_t x, uint32_t y) {
    // Row-or-column projection of the admitted (C,1,1) cluster, with one MMA group.
    uint16_t mask = 0;
    for (int peer = 0; peer < ClusterSize; ++peer)
      if (peer / ClusterSize == x / ClusterSize || y == 0) mask |= uint16_t(1u << peer);
    return mask;
  }

  __host__ __device__ static constexpr uint16_t w8BMask(uint32_t y) { return y == 0 ? 3 : 0; }

  MSCCLPP_DEVICE_INLINE void initW4Mask(dim3 block) {
    if (role_ == Role::Consumer) mask_ = w4Mask(block.x, block.y);
  }
  MSCCLPP_DEVICE_INLINE void initW8BMask() {
    static_assert(ClusterSize == 2);
    mask_ = w8BMask(moeClusterBlock().y);
  }

  MSCCLPP_DEVICE_INLINE MoeStageReady producer_try_acquire(PipelineState s, uint32_t skip) {
    stageCheckRole(role_, Role::Producer);
    if (skip) return MoeStageReady::WaitDone;
    return static_cast<MoeStageReady>(stageTryWait(&empty_[s.index()], s.phase()));
  }

  MSCCLPP_DEVICE_INLINE void producer_acquire(PipelineState s, MoeStageReady ready = MoeStageReady::WaitAgain) {
    stageCheckRole(role_, Role::Producer);
    if (ready != MoeStageReady::WaitDone) stageWait(&empty_[s.index()], s.phase());
  }

  MSCCLPP_DEVICE_INLINE void producer_tail(PipelineState s) {
    stageCheckRole(role_, Role::Producer);
    for (int i = 0; i < Slots; ++i) {
      stageWait(&empty_[s.index()], s.phase());
      ++s;
    }
  }

  MSCCLPP_DEVICE_INLINE uint64_t* producer_get_barrier(PipelineState s) { return &full_[s.index()]; }

  MSCCLPP_DEVICE_INLINE MoeStageReady consumer_try_wait(PipelineState s, uint32_t skip) {
    stageCheckRole(role_, Role::Consumer);
    if (skip) return MoeStageReady::WaitDone;
    return static_cast<MoeStageReady>(stageTryWait(&full_[s.index()], s.phase()));
  }

  MSCCLPP_DEVICE_INLINE void consumer_wait(PipelineState s, MoeStageReady ready = MoeStageReady::WaitAgain) {
    stageCheckRole(role_, Role::Consumer);
    if (ready == MoeStageReady::WaitAgain) stageWait(&full_[s.index()], s.phase());
  }

  MSCCLPP_DEVICE_INLINE void consumer_release(PipelineState s) {
    stageCheckRole(role_, Role::Consumer);
    stageCommit<ClusterSize>(&empty_[s.index()], mask_);
  }

 private:
  uint64_t* full_;
  uint64_t* empty_;
  Role role_;
  uint16_t mask_ = 0;
};

// W8 FP8 A+scale: each CTA's leader publishes its own byte count; one
// selected lane in the local transform warpgroup releases EMPTY to that CTA.
template <int Slots>
struct MoeW8WeightPipe {
  using SharedStorage = MoeStageStorage<Slots>;
  using PipelineState = MoeStageState<Slots>;
  using Role = MoeStageRole;

  MSCCLPP_DEVICE_INLINE MoeW8WeightPipe(SharedStorage& storage, Role role, uint32_t leader,
                                       uint32_t bytes, int initializingWarp)
      : full_(storage.full_barrier_), empty_(storage.empty_barrier_), role_(role), leader_(leader), bytes_(bytes) {
    if (stageWarp() == initializingWarp) stageInitPair(storage, 1, 1);
    fenceBarrierInit();
  }

  // Selected Swizzle<2,0,-2> of the four-by-four warpgroup release layout.
  __host__ __device__ static constexpr uint32_t arrivalTarget(uint32_t warp, uint32_t thread) {
    const uint32_t linear = (warp % 4) * 4 + (thread / 8) % 4;
    return linear ^ ((linear & 3u) << 2);
  }
  __host__ __device__ static constexpr uint32_t signals(uint32_t warp, uint32_t thread, uint32_t cta) {
    const uint32_t dst = arrivalTarget(warp, thread);
    return (thread % 8 == 0) & (dst < 2) & ((dst % 2 == cta) && (dst % 2 == cta % 2));
  }

  MSCCLPP_DEVICE_INLINE void init_masks(dim3 block) {
    if (role_ == Role::Consumer) {
      const int warp = stageWarp();
      const int thread = threadIdx.x % 128;
      target_ = arrivalTarget(warp, thread);
      signaling_ = signals(warp, thread, block.x);
    }
  }

  MSCCLPP_DEVICE_INLINE MoeStageReady producer_try_acquire(PipelineState s, uint32_t skip) {
    stageCheckRole(role_, Role::Producer);
    if (skip) return MoeStageReady::WaitDone;
    return static_cast<MoeStageReady>(stageTryWait(&empty_[s.index()], s.phase()));
  }

  MSCCLPP_DEVICE_INLINE void producer_acquire(PipelineState s, MoeStageReady ready) {
    stageCheckRole(role_, Role::Producer);
    if (ready != MoeStageReady::WaitDone) stageWait(&empty_[s.index()], s.phase());
    if (leader_) arriveMoeTmaExpect(&full_[s.index()], bytes_);
#ifndef NDEBUG
    if (leader_ && threadIdx.x % WARP_SIZE != 0) asm volatile("brkpt;" ::);
#endif
  }

  MSCCLPP_DEVICE_INLINE void producer_tail(PipelineState s) {
    stageCheckRole(role_, Role::Producer);
    for (int i = 0; i < Slots; ++i) {
      stageWait(&empty_[s.index()], s.phase());
      ++s;
    }
  }

  MSCCLPP_DEVICE_INLINE uint64_t* producer_get_barrier(PipelineState s) { return &full_[s.index()]; }

  MSCCLPP_DEVICE_INLINE void consumer_wait(PipelineState s) {
    stageCheckRole(role_, Role::Consumer);
    stageWait(&full_[s.index()], s.phase());
  }

  MSCCLPP_DEVICE_INLINE void consumer_release(PipelineState s) {
    stageCheckRole(role_, Role::Consumer);
    stageArriveCluster(&empty_[s.index()], target_, signaling_);
  }

 private:
  uint64_t* full_;
  uint64_t* empty_;
  Role role_;
  uint32_t leader_;
  uint32_t bytes_;
  uint32_t target_ = 0;
  uint32_t signaling_ = 0;
};

// W8 converted A: all 128 transform lanes in each CTA announce their
// fenced store to CTA0's FULL. One MMA commit releases EMPTY to both CTAs.
template <int Slots>
struct MoeW8TransformPipe {
  using SharedStorage = MoeStageStorage<Slots>;
  using PipelineState = MoeStageState<Slots>;
  using Role = MoeStageRole;

  __host__ __device__ static constexpr uint32_t producerArrivals() { return 2 * 128; }

  MSCCLPP_DEVICE_INLINE MoeW8TransformPipe(SharedStorage& storage, Role role, int initializingWarp)
      : full_(storage.full_barrier_), empty_(storage.empty_barrier_), role_(role) {
    if (stageWarp() == initializingWarp) stageInitPair(storage, producerArrivals(), 1);
    fenceBarrierInit();
  }

  MSCCLPP_DEVICE_INLINE void init_masks(dim3 block = moeClusterBlock()) {
    if (role_ == Role::Consumer) mask_ = stagePairMask(block.x);
  }

  MSCCLPP_DEVICE_INLINE void producer_acquire(PipelineState s) {
    stageCheckRole(role_, Role::Producer);
    stageWait(&empty_[s.index()], s.phase());
  }

  MSCCLPP_DEVICE_INLINE void producer_commit(PipelineState s) {
    stageCheckRole(role_, Role::Producer);
    stageArriveLeader(&full_[s.index()]);
  }

  MSCCLPP_DEVICE_INLINE void producer_tail(PipelineState s) {
    for (int i = 0; i < Slots; ++i) {
      producer_acquire(s);
      ++s;
    }
  }

  MSCCLPP_DEVICE_INLINE MoeStageReady consumer_try_wait(PipelineState s, uint32_t skip) {
    stageCheckRole(role_, Role::Consumer);
    if (skip) return MoeStageReady::WaitDone;
    return static_cast<MoeStageReady>(stageTryWait(&full_[s.index()], s.phase()));
  }

  MSCCLPP_DEVICE_INLINE void consumer_wait(PipelineState s, MoeStageReady ready) {
    if (ready == MoeStageReady::WaitAgain) {
      stageCheckRole(role_, Role::Consumer);
      stageWait(&full_[s.index()], s.phase());
    }
  }

  MSCCLPP_DEVICE_INLINE void consumer_release(PipelineState s) {
    stageCheckRole(role_, Role::Consumer);
    stageCommit<2>(&empty_[s.index()], mask_);
  }

 private:
  uint64_t* full_;
  uint64_t* empty_;
  Role role_;
  uint16_t mask_ = 0;
};

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_LOAD_PIPES_CUH_
