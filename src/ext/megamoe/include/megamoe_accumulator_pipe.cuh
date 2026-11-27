// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Selected accumulator ring from CUTLASS 147295a3d4b75f3aeff247c25b8927cea9a7006a,
// cutlass/pipeline/{sm100_pipeline,sm90_pipeline}.hpp. See ../LICENSE.cutlass.
// Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MSCCLPP_EXT_MEGAMOE_ACCUMULATOR_PIPE_CUH_
#define MSCCLPP_EXT_MEGAMOE_ACCUMULATOR_PIPE_CUH_

#include <cassert>
#include "megamoe_stage_sync.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

// The MMA producer waits for EMPTY and commits FULL; all epilogue lanes
// wait for FULL and release EMPTY back to the leader CTA.
template <int Slots, int ClusterSize>
struct MoeAccumulatorPipeline {
  static_assert(Slots == 2 || Slots == 3);
  static_assert(ClusterSize == 1 || ClusterSize == 2);
  static constexpr int Stages = Slots;
  using PipelineState = MoeStageState<Slots>;
  using ThreadCategory = MoeStageRole;
  using SharedStorage = MoeStageStorage<Slots>;
  struct Params {
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t producer_arv_count = 1;
    uint32_t consumer_arv_count = 1;
    uint32_t dst_blockid = moeClusterRank();
    int initializing_warp = 0;
  };

  MSCCLPP_DEVICE_INLINE MoeAccumulatorPipeline(SharedStorage& storage, Params params)
      : params_(params), full_(storage.full_barrier_), empty_(storage.empty_barrier_) {
    if (stageWarp() == params.initializing_warp) {
      assert(params.producer_arv_count > 0 && params.consumer_arv_count > 0);
      stageInitPair(storage, params.producer_arv_count, params.consumer_arv_count);
    }
    fenceBarrierInit();
  }

  __host__ __device__ static constexpr uint32_t consumerArrivals(uint32_t epilogueWarps) {
    return ClusterSize * epilogueWarps * WARP_SIZE;
  }

  __host__ __device__ static constexpr uint16_t peerMask(uint32_t x) {
    if constexpr (ClusterSize == 2) return stagePairMask(x);
    else return uint16_t(1u << x);
  }

  MSCCLPP_DEVICE_INLINE void init_masks(dim3 block = moeClusterBlock()) {
    if (params_.role == ThreadCategory::Producer) mask_ = peerMask(block.x);
  }

  MSCCLPP_DEVICE_INLINE void producer_acquire(PipelineState state) {
    stageCheckRole(params_.role, ThreadCategory::Producer);
    stageWait(&empty_[state.index()], state.phase());
  }

  MSCCLPP_DEVICE_INLINE void producer_commit(PipelineState state) {
    stageCheckRole(params_.role, ThreadCategory::Producer);
    stageCommit<ClusterSize>(&full_[state.index()], mask_);
  }

  MSCCLPP_DEVICE_INLINE void producer_tail(PipelineState state) {
    for (int count = 0; count < Slots; ++count) {
      producer_acquire(state);
      ++state;
    }
  }

  MSCCLPP_DEVICE_INLINE void consumer_wait(PipelineState state) {
    stageCheckRole(params_.role, ThreadCategory::Consumer);
    stageWait(&full_[state.index()], state.phase());
  }

  MSCCLPP_DEVICE_INLINE void consumer_release(PipelineState state) {
    stageCheckRole(params_.role, ThreadCategory::Consumer);
    if constexpr (ClusterSize == 2) stageArriveLeader(&empty_[state.index()]);
    else stageArriveCluster(&empty_[state.index()], params_.dst_blockid, 1u);
  }

 private:
  Params params_;
  uint64_t* full_;
  uint64_t* empty_;
  uint16_t mask_ = 0;
};

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_ACCUMULATOR_PIPE_CUH_
