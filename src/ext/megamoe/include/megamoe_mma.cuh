// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_MMA_CUH_
#define MSCCLPP_EXT_MEGAMOE_MMA_CUH_

#include "megamoe_collective.cuh"
#include "megamoe_tasks.cuh"
#include "megamoe_trace.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <class Pipeline, class Schedule>
MSCCLPP_DEVICE_INLINE Pipeline makeAccumulatorPipeline(typename Pipeline::SharedStorage& storage, int warp) {
  typename Pipeline::Params params{};
  params.role = warp == Schedule::MmaWarp ? Pipeline::ThreadCategory::Producer
                                          : (warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd
                                                 ? Pipeline::ThreadCategory::Consumer
                                                 : Pipeline::ThreadCategory::NonParticipant);
  params.producer_arv_count = 1;
  params.consumer_arv_count = Pipeline::consumerArrivals(Schedule::EpilogueEnd - Schedule::EpilogueBegin);
  params.initializing_warp = Schedule::EpilogueBegin;
  return Pipeline(storage, params);
}

MSCCLPP_DEVICE_INLINE void issueW8A16Mma(uint32_t destination, uint32_t weightAddress, uint64_t activationDescriptor,
                                         uint32_t instruction, uint32_t accumulate) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %4, 0;"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3,"
      "{%5,%5,%5,%5,%5,%5,%5,%5}, p; }"
      :
      : "r"(destination), "r"(weightAddress), "l"(activationDescriptor), "r"(instruction), "r"(accumulate),
        "r"(uint32_t(0)));
}

template <int ClusterSize>
MSCCLPP_DEVICE_INLINE void issueW4A8BlockScaledMma(uint32_t destination, uint64_t descriptorA, uint64_t descriptorB,
                                                   uint32_t instruction, uint32_t scaleA, uint32_t scaleB,
                                                   uint32_t accumulate) {
  static_assert(ClusterSize == 1 || ClusterSize == 2);
  if constexpr (ClusterSize == 1) {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %6, 0;"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], p; }"
        :
        : "r"(destination), "l"(descriptorA), "l"(descriptorB), "r"(instruction), "r"(scaleA), "r"(scaleB),
          "r"(accumulate));
  } else {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %6, 0;"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], p; }"
        :
        : "r"(destination), "l"(descriptorA), "l"(descriptorB), "r"(instruction), "r"(scaleA), "r"(scaleB),
          "r"(accumulate));
  }
}

// Selected UTCCP issue from CUTLASS 147295a3d4b75f3aeff247c25b8927cea9a7006a
// (cute/arch/copy_sm100.hpp). One broadcast copy per K128, in ascending K order.
// Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause; see ../LICENSE.cutlass.
template <int ClusterSize, int TileK>
MSCCLPP_DEVICE_INLINE void copyW4Scale(uint32_t source, uint32_t target, int stage) {
  static_assert(ClusterSize == 1 || ClusterSize == 2);
#pragma unroll
  for (int i = 0; i < TileK / 128; ++i) {
    const uint64_t desc = w4UtccpSource<TileK>(source, stage, i);
    const uint32_t tmem = target + i * 4;
    if constexpr (ClusterSize == 1)
      asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(tmem), "l"(desc));
    else
      asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" ::"r"(tmem), "l"(desc));
  }
}

template <bool Local>
struct W8MmaStates {
  using Types = W8A16CollectiveTypesT<Local>;
  typename Types::LoadB::PipelineState load;
  typename Types::Transform::PipelineState transform;
  typename Types::Accumulate::PipelineState accumulate;
};

template <bool Local = false>
MSCCLPP_DEVICE_INLINE W8MmaStates<Local> mmaW8A16Tiles(
    typename W8A16CollectiveTypesT<Local>::LoadB loadPipeline,
    typename W8A16CollectiveTypesT<Local>::LoadB::PipelineState loadState,
    typename W8A16CollectiveTypesT<Local>::Transform transformPipeline,
    typename W8A16CollectiveTypesT<Local>::Transform::PipelineState transformState,
    typename W8A16CollectiveTypesT<Local>::Accumulate accumulatePipeline,
    typename W8A16CollectiveTypesT<Local>::Accumulate::PipelineState accumulateState,
    uint32_t accumulators, uint32_t weightBase, uint32_t activationBase, int kTiles, int validRows) {
  auto currentLoad = loadState;
  auto nextLoad = loadState;
  auto currentTransform = transformState;
  auto nextTransform = transformState;
  uint32_t skip = kTiles <= 0;
  auto transformReady = transformPipeline.consumer_try_wait(nextTransform, skip);
  auto loadReady = loadPipeline.consumer_try_wait(nextLoad, skip);
  ++nextTransform;
  ++nextLoad;
  accumulatePipeline.producer_acquire(accumulateState);
  const uint32_t destination = accumulatorStage(accumulators, accumulateState.index());
  auto currentAccumulate = accumulateState;
  ++accumulateState;
  uint32_t accumulate = 0;
  bool issuer = electMoeLane();
  const uint32_t instruction = w8Instruction(validRows);
#pragma unroll 1
  for (; kTiles > 0; --kTiles) {
    loadPipeline.consumer_wait(currentLoad, loadReady);
    transformPipeline.consumer_wait(currentTransform, transformReady);
    // Elect once for K128. Wait/release still require the complete MMA warp.
    if (issuer) {
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        issueW8A16Mma(destination, w8WeightAddress(weightBase, currentTransform.index(), k),
                      w8ActivationDescriptor(activationBase, currentLoad.index(), k), instruction,
                      k == 0 ? accumulate : 1u);
      }
    }
    accumulate = 1;
    loadPipeline.consumer_release(currentLoad);
    transformPipeline.consumer_release(currentTransform);
    skip = kTiles <= 1;
    loadReady = loadPipeline.consumer_try_wait(nextLoad, skip);
    transformReady = transformPipeline.consumer_try_wait(nextTransform, skip);
    currentLoad = nextLoad;
    currentTransform = nextTransform;
    ++nextLoad;
    ++nextTransform;
  }
  accumulatePipeline.producer_commit(currentAccumulate);
  return {currentLoad, currentTransform, accumulateState};
}

template <class KernelParams, class LoadPipeline, class LoadState, class AccumulatePipeline,
          class AccumulateState, class MmaInputs>
MSCCLPP_DEVICE_INLINE LoadState mmaW4A8Tiles(LoadPipeline& loadPipe, LoadPipeline& activationPipe, LoadState loadState,
                                             AccumulatePipeline& accPipe, AccumulateState accState,
                                             uint32_t destination, const MmaInputs& inputs, int kTiles,
                                             int validRows) {
  using Types = typename KernelParams::Collective;
  constexpr int ClusterSize = KernelParams::Tiles::ClusterSize;
  constexpr int TileK = KernelParams::Tiles::K;
  constexpr bool SplitPipelines = Types::SplitPipelines;
  traceKernel(KernelTracePhase::AccumulatorFree, true, accState.index());
  accPipe.producer_acquire(accState);
  traceKernel(KernelTracePhase::AccumulatorFree, false, accState.index());
  uint32_t accumulate = 0;
  uint64_t waitNs = 0;
  uint64_t activationWaitNs = 0;
  uint64_t issueNs = 0;
  uint64_t releaseNs = 0;
  uint32_t weightReady = 0;
  uint32_t activationReady = 0;
  const int iterations = kTiles;
  // Elect once for the entire K-tile loop while keeping pipeline waits/releases warp-wide.
  const bool issuer = electMoeLane();
  for (; kTiles > 0; --kTiles) {
#if MSCCLPP_MEGAMOE_TRACE == 1
    // Nonblocking probes sample both operands before either blocking wait.
    weightReady += stageTestWait(loadPipe.producer_get_barrier(loadState), loadState.phase());
    if constexpr (SplitPipelines)
      activationReady += stageTestWait(activationPipe.producer_get_barrier(loadState), loadState.phase());
#endif
    auto start = traceKernelClock();
    loadPipe.consumer_wait(loadState);
    waitNs += traceKernelClock() - start;
    if constexpr (SplitPipelines) {
      start = traceKernelClock();
      activationPipe.consumer_wait(loadState);
      activationWaitNs += traceKernelClock() - start;
    }
    int stage = loadState.index();
    start = traceKernelClock();
    if (issuer) {
      copyW4Scale<ClusterSize, TileK>(inputs.sharedScaleA, inputs.scaleA, stage);
      if constexpr (KernelParams::Activation == ActivationDtype::Mxfp8)
        copyW4Scale<ClusterSize, TileK>(inputs.sharedScaleB, inputs.scaleB, stage);
#pragma unroll
      for (int k = 0; k < TileK / 32; ++k) {
        const uint32_t scaleA = w4MmaScale(inputs.scaleA, k);
        const uint32_t scaleB = w4MmaScale(inputs.scaleB, k);
        const uint32_t instruction = w4Instruction<ClusterSize>(validRows, scaleA, scaleB);
        issueW4A8BlockScaledMma<ClusterSize>(destination, w4MmaData<TileK, 128>(inputs.dataA, stage, k),
                                             w4MmaData<TileK, 128 / ClusterSize>(inputs.dataB, stage, k),
                                             instruction, scaleA, scaleB, k == 0 ? accumulate : 1u);
      }
    }
    accumulate = 1;
    issueNs += traceKernelClock() - start;
    start = traceKernelClock();
    loadPipe.consumer_release(loadState);
    if constexpr (SplitPipelines) activationPipe.consumer_release(loadState);
    releaseNs += traceKernelClock() - start;
    ++loadState;
  }
  if constexpr (SplitPipelines) {
    traceKernelTotal(KernelTracePhase::MmaWeightWait, waitNs, iterations);
    traceKernelTotal(KernelTracePhase::MmaActivationWait, activationWaitNs, iterations);
    traceKernelCount(KernelTracePhase::WeightReadyAtEntry, weightReady, iterations);
    traceKernelCount(KernelTracePhase::ActivationReadyAtEntry, activationReady, iterations);
  } else {
    traceKernelTotal(KernelTracePhase::MmaInputWait, waitNs, iterations);
    traceKernelCount(KernelTracePhase::InputReadyAtEntry, weightReady, iterations);
  }
  traceKernelTotal(KernelTracePhase::MmaIssue, issueNs, iterations);
  traceKernelTotal(KernelTracePhase::StageRelease, releaseNs, iterations);
  return loadState;
}

template <bool Local, class Schedule, class KernelParams, class Storage, class LoadB, class Transform, class Accumulate>
MSCCLPP_DEVICE_INLINE void runW8A16MmaRole(const KernelParams& params, int nTokens, Storage& sharedStorage,
                                           LoadB& bPipeline, Transform& tPipeline, Accumulate& cPipeline,
                                           uint32_t accumulators, int hidden, int intermediate, int warpIdx,
                                           int ctaIdx, int clusterIdx, int nTasks, int nTokenTiles, int nLocalExperts) {
  if (warpIdx == Schedule::MmaWarp && ctaIdx == 0) {
    typename LoadB::PipelineState bState;
    typename Transform::PipelineState tState;
    auto cState = producerStart<Accumulate>();
    const uint32_t weightBase = W8A16Mainloop::w8WeightTmemBase(accumulators);
    const auto activationBase = uint32_t(__cvta_generic_to_shared(sharedStorage.tensors.input.smem_B));
    const auto scheduler = makeTaskScheduler<KernelParams>(nTokenTiles, nLocalExperts, hidden, intermediate);
    const int numClusters = gridDim.x / KernelParams::Tiles::ClusterSize;
    if (clusterIdx < nTasks) {
      Task nextTask = scheduler.taskAt(params, clusterIdx, nTokens);
      for (int i = clusterIdx; i < nTasks; i += numClusters) {
        Task task = nextTask;
        const bool hasNext = (i + numClusters) < nTasks;
        auto phase = task.fc1 ? KernelTracePhase::MmaFc1 : KernelTracePhase::MmaFc2;
        traceKernel(phase, true, i);
        auto result = mmaW8A16Tiles<Local>(bPipeline, bState, tPipeline, tState, cPipeline, cState, accumulators,
                                           weightBase, activationBase, taskKTiles(params, task, hidden, intermediate),
                                           task.tokens.nTokens);
        if (hasNext) nextTask = scheduler.taskAt(params, i + numClusters, nTokens);
        traceKernel(phase, false, i);
        bState = result.load;
        tState = result.transform;
        cState = result.accumulate;
      }
    }
    cPipeline.producer_tail(cState);
  }
}

template <class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE void runMmaRole(const KernelParams& params, Storage& sharedStorage,
                                      typename KernelParams::Collective::Load& loadPipeline,
                                      typename KernelParams::Collective::Load& activationPipeline,
                                      typename KernelParams::Collective::Accumulate& accumulatePipeline,
                                      uint32_t accumulators, int hidden, int intermediate, int warpIdx, int ctaIdx,
                                      int clusterIdx, int nTasks, int nTokenTiles, int nLocalExperts) {
  using Types = typename KernelParams::Collective;
  using Mainloop = typename Types::Mainloop;
  using Load = typename Types::Load;
  using Accumulate = typename Types::Accumulate;
  using Schedule = typename Types::WarpSchedule;
  if (warpIdx == Schedule::MmaWarp && ctaIdx == 0) {
    (void)ctaIdx;
    typename Load::PipelineState loadState;
    auto accumulateState = producerStart<Accumulate>();
    auto mmaInputs = Mainloop::mma_init(accumulators, sharedStorage.tensors);
    if constexpr (KernelParams::Activation == ActivationDtype::Fp8Static) {
      if (electMoeLane()) {
        copyW4Scale<KernelParams::Tiles::ClusterSize, KernelParams::Tiles::K>(mmaInputs.sharedScaleB,
                                                                              mmaInputs.scaleB, 0);
      }
    }
    const auto scheduler = makeTaskScheduler<KernelParams>(nTokenTiles, nLocalExperts, hidden, intermediate);
    const int numClusters = gridDim.x / KernelParams::Tiles::ClusterSize;
    if (clusterIdx < nTasks) {
      Task nextTask = scheduler.taskAt(params, clusterIdx);
      for (int i = clusterIdx; i < nTasks; i += numClusters) {
        Task task = nextTask;
        const bool hasNext = (i + numClusters) < nTasks;
        int kTiles = taskKTiles(params, task, hidden, intermediate);
        const uint32_t accumulator = accumulatorStage(accumulators, accumulateState.index());
        auto phase = task.fc1 ? KernelTracePhase::MmaFc1 : KernelTracePhase::MmaFc2;
        traceKernel(phase, true, i);
        loadState = mmaW4A8Tiles<KernelParams>(
            loadPipeline, activationPipeline, loadState, accumulatePipeline, accumulateState, accumulator, mmaInputs,
            kTiles, task.tokens.nTokens);
        if (hasNext) nextTask = scheduler.taskAt(params, i + numClusters);
        traceKernel(phase, false, i);
        accumulatePipeline.producer_commit(accumulateState);
        ++accumulateState;
      }
    }
    accumulatePipeline.producer_tail(accumulateState);
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_MMA_CUH_
