// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_W8A16_ROLES_CUH_
#define MSCCLPP_EXT_MEGAMOE_W8A16_ROLES_CUH_

#include "megamoe_collective.cuh"
#include "megamoe_combine.cuh"
#include "megamoe_dispatch.cuh"
#include "megamoe_epilogue.cuh"
#include "megamoe_mma.cuh"
#include "megamoe_quantization.cuh"
#include "megamoe_routing.cuh"
#include "megamoe_tasks.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <bool Local>
struct W8A16Pipelines {
  using Types = W8A16CollectiveTypesT<Local>;
  typename Types::LoadA loadA;
  typename Types::LoadB loadB;
  typename Types::Transform transform;
  typename Types::Accumulate accumulate;
  MegaMoeTmemAllocator<2> allocator;
};

template <bool Local, class Storage>
MSCCLPP_DEVICE_INLINE W8A16Pipelines<Local> initW8A16PipelinesAndStorage(Storage& sharedStorage,
                                                                         const KernelThreadLocation& threadLoc) {
  using Types = W8A16CollectiveTypesT<Local>;
  using Mainloop = typename Types::Mainloop;
  using LoadA = typename Types::LoadA;
  using LoadB = typename Types::LoadB;
  using Transform = typename Types::Transform;
  using Accumulate = typename Types::Accumulate;
  using Schedule = typename Types::WarpSchedule;

  const bool transformWarp =
      threadLoc.warpIdx >= Schedule::TransformBegin && threadLoc.warpIdx < Schedule::TransformEnd;
  const bool mmaWarp = threadLoc.warpIdx == Schedule::MmaWarp;
  const auto aRole = threadLoc.warpIdx == Schedule::LoadAWarp
                         ? MoeStageRole::Producer
                         : (transformWarp ? MoeStageRole::Consumer : MoeStageRole::NonParticipant);
  LoadA aPipeline(sharedStorage.mainloopA, aRole, threadLoc.laneIdx == 0, Mainloop::TmaTransactionBytes_A,
                  Schedule::LoadAWarp);

  const auto bRole = threadLoc.warpIdx == Schedule::LoadBWarp
                         ? MoeStageRole::Producer
                         : (mmaWarp ? MoeStageRole::Consumer : MoeStageRole::NonParticipant);
  // Only load_B publishes the exact per-subtile transaction bytes for B.
  LoadB bPipeline(sharedStorage.mainloopB, bRole);
  LoadB::init(sharedStorage.mainloopB, Schedule::LoadBWarp);

  const auto tRole = transformWarp ? MoeStageRole::Producer
                                  : (mmaWarp ? MoeStageRole::Consumer : MoeStageRole::NonParticipant);
  Transform tPipeline(sharedStorage.transformed, tRole, Schedule::TransformBegin);
  auto cPipeline = makeAccumulatorPipeline<Accumulate, Schedule>(sharedStorage.accumulated, threadLoc.warpIdx);
  aPipeline.init_masks(moeClusterBlock());
  bPipeline.initW8BMask();
  tPipeline.init_masks();
  cPipeline.init_masks();
  MegaMoeTmemAllocator<2> allocator{};
  if (threadLoc.warpIdx == Schedule::MmaWarp) allocator.allocate(512, &sharedStorage.tmem);
  if constexpr (!Local)
    initDispatchStorage<Types::DispatchStages, Schedule>(sharedStorage, threadLoc.warpIdx, threadLoc.laneIdx);
  fenceBarrierInit();
  return W8A16Pipelines<Local>{aPipeline, bPipeline, tPipeline, cPipeline, allocator};
}

template <bool Local, class KernelParams>
MSCCLPP_DEVICE_INLINE int runW8A16RoutingStage(const KernelParams& params, int nTokens, const int32_t* ids,
                                               const float* scores) {
  if constexpr (Local) {
    __syncthreads();
    clusterSync();
    return (nTokens + KernelParams::Tiles::N - 1) / KernelParams::Tiles::N;
  } else {
    traceKernel(KernelTracePhase::Routing, true);
    if (params.config.computeOnly) {
      prepareLocalRoutes(params, nTokens, ids);
    } else {
      prepareRoutes<false>(params, nTokens, ids, scores);
    }
    traceKernel(KernelTracePhase::Routing, false);
    return params.workspace.control->nTokenTiles;
  }
}

// One lane reads one SW128 row. All SMEM reads finish before the stage is
// released. Converted K16 groups then store in ascending order to TMEM.
// sm100 store opcode and register order from pinned cute/arch/copy_sm100.hpp.
template <bool Local = false>
MSCCLPP_DEVICE_INLINE void transformWeights(
    typename W8A16CollectiveTypesT<Local>::LoadA& load,
    typename W8A16CollectiveTypesT<Local>::LoadA::PipelineState& loadState,
    typename W8A16CollectiveTypesT<Local>::Transform& transformed,
    typename W8A16CollectiveTypesT<Local>::Transform::PipelineState& transformState,
    const W8A16Mainloop::TensorStorage& shared, uint32_t weightBase, int kTiles) {
  const int lane = threadIdx.x % 128;
#pragma unroll 1
  for (int k = 0; k < kTiles; ++k) {
    load.consumer_wait(loadState);
    transformed.producer_acquire(transformState);
    const int stage = loadState.index();
    uint4 raw[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      raw[i] = *reinterpret_cast<const uint4*>(shared.input.smem_A + w8RawOffset(stage, lane, i * 16));
    uint8_t blockScales[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) blockScales[i] = shared.input.smem_scale[w8ScaleOffset(stage, lane, i)];
    namedBarrierSync(128, TransformBarrier);
    load.consumer_release(loadState);
    ++loadState;

    uint32_t converted[8][8];
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const uint32_t expanded = expandE8M0ScalePair(blockScales[b]);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int i = b * 2 + half;
        const uint32_t words[4] = {raw[i].x, raw[i].y, raw[i].z, raw[i].w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          converted[i][2 * j] = decodeScaledPairWithExpandedScale(uint16_t(words[j]), expanded);
          converted[i][2 * j + 1] = decodeScaledPairWithExpandedScale(uint16_t(words[j] >> 16), expanded);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const auto& x = converted[i];
      const uint32_t dst = w8WeightAddress(weightBase, transformState.index(), i, lane);
      asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
                   :: "r"(dst), "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]),
                      "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]));
    }
    waitTmemStore();
    transformed.producer_commit(transformState);
    ++transformState;
  }
}

template <bool Local, class KernelParams, class Storage, class LoadA, class Transform>
MSCCLPP_DEVICE_INLINE void runW8A16TransformRole(const KernelParams& params, int nTokens, Storage& sharedStorage,
                                                 LoadA& loadPipeline, Transform& transformPipeline,
                                                 uint32_t accumulators, int hidden, int intermediate,
                                                 int clusterIdx, int nTasks, int nTokenTiles, int nLocalExperts) {
  using Mainloop = typename KernelParams::Collective::Mainloop;
  warpgroupRegAlloc<KernelParams::Collective::TransformRegisters>();
  typename LoadA::PipelineState loadState;
  auto transformState = producerStart<Transform>();
  const uint32_t weightBase = Mainloop::w8WeightTmemBase(accumulators);
  const auto scheduler = makeTaskScheduler<KernelParams>(nTokenTiles, nLocalExperts, hidden, intermediate);
  const int numClusters = gridDim.x / KernelParams::Tiles::ClusterSize;
  if (clusterIdx < nTasks) {
    Task nextTask = scheduler.taskAt(params, clusterIdx, nTokens);
    for (int i = clusterIdx; i < nTasks; i += numClusters) {
      Task task = nextTask;
      const bool hasNext = (i + numClusters) < nTasks;
      auto phase = task.fc1 ? KernelTracePhase::TransformFc1 : KernelTracePhase::TransformFc2;
      traceKernel(phase, true, i);
      transformWeights<Local>(loadPipeline, loadState, transformPipeline, transformState, sharedStorage.tensors,
                              weightBase, taskKTiles(params, task, hidden, intermediate));
      if (hasNext) nextTask = scheduler.taskAt(params, i + numClusters, nTokens);
      traceKernel(phase, false, i);
    }
  }
  transformPipeline.producer_tail(transformState);
}

template <bool Local, class KernelParams, class LoadTiles>
MSCCLPP_DEVICE_INLINE void loadW8A16ActivationTiles(const KernelParams& params, const Task& task, int hidden,
                                                    int intermediate, int kTiles, LoadTiles&& loadTiles) {
  using Types = typename KernelParams::Collective;
  if constexpr (Local) {
    if (!task.fc1)
      waitReadyCounter(params.workspace.hiddenReady + size_t(task.block) * W4ReadyCounterStride,
                       fc1CompletionCount<KernelParams>(intermediate));
    loadTiles(0, kTiles);
    return;
  }
  if (task.fc1) {
    constexpr int TilesPerChunk = Types::DispatchChunk / (KernelParams::Tiles::K * sizeof(__bfloat16));
    for (int first = 0; first < kTiles; first += TilesPerChunk) {
      traceKernel(KernelTracePhase::TokenReady, true, first / TilesPerChunk);
      waitReadyCounter(w4InputChunkCounter<Types>(params.workspace, hidden, task.block, first / TilesPerChunk),
                       task.tokens.nTokens);
      traceKernel(KernelTracePhase::TokenReady, false, first / TilesPerChunk);
      loadTiles(first, min(TilesPerChunk, kTiles - first));
    }
  } else {
    traceKernel(KernelTracePhase::TokenReady, true, int(task.fc1));
    waitReadyCounter(params.workspace.hiddenReady + size_t(task.block) * W4ReadyCounterStride,
                     fc1CompletionCount<KernelParams>(intermediate));
    traceKernel(KernelTracePhase::TokenReady, false, int(task.fc1));
    loadTiles(0, kTiles);
  }
}

template <class KernelParams>
struct W8A16ComputeContext {
  using Types = typename KernelParams::Collective;
  using Mainloop = typename Types::Mainloop;
  Mainloop fc1;
  Mainloop fc2;
  uint32_t accumulators;
  int nTokenTiles;
  int nTasks;
  int nLocalExperts;
};

template <class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE W8A16ComputeContext<KernelParams> makeW8A16ComputeContext(const KernelParams& params,
                                                                                Storage& sharedStorage, int ctaIdx,
                                                                                int nTokenTiles) {
  using Mainloop = typename KernelParams::Collective::Mainloop;
  const int hidden = params.config.hidden;
  const int intermediate = params.config.intermediate;
  const int nTasks = nTokenTiles * (fc1TaskTiles<KernelParams>(intermediate) + fc2TaskTiles<KernelParams>(hidden));
  const int nLocalExperts = params.config.numExperts / params.config.worldSize;
  return {Mainloop(ctaIdx), Mainloop(ctaIdx), sharedStorage.tmem, nTokenTiles, nTasks, nLocalExperts};
}

template <class KernelParams, class Storage, class LoadA, class ComputeCtx>
MSCCLPP_DEVICE_INLINE void runW8A16LoadAStage(const KernelParams& params, Storage& sharedStorage, LoadA& aPipeline,
                                              ComputeCtx& computeCtx, const KernelThreadLocation& threadLoc,
                                              int nTokens) {
  if (threadLoc.laneIdx != 0) return;
  const int hidden = params.config.hidden;
  const int intermediate = params.config.intermediate;
  auto state = producerStart<LoadA>();
  computeCtx.fc1.prefetch(params.fc1);
  computeCtx.fc2.prefetch(params.fc2);
  const auto scheduler =
      makeTaskScheduler<KernelParams>(computeCtx.nTokenTiles, computeCtx.nLocalExperts, hidden, intermediate);
  const int numClusters = gridDim.x / KernelParams::Tiles::ClusterSize;
  if (threadLoc.clusterIdx < computeCtx.nTasks) {
    Task nextTask = scheduler.taskAt(params, threadLoc.clusterIdx, nTokens);
    for (int i = threadLoc.clusterIdx; i < computeCtx.nTasks; i += numClusters) {
      Task task = nextTask;
      const bool hasNext = (i + numClusters) < computeCtx.nTasks;
      auto coord = taskCoord(params, task, threadLoc.ctaIdx);
      int kTiles = taskKTiles(params, task, hidden, intermediate);
      const bool weightEvictNormal =
          KernelParams::LocalExpert ? (nTokens > KernelParams::Tiles::N) : task.tokens.weightEvictNormal();
      auto phase = task.fc1 ? KernelTracePhase::LoadFc1 : KernelTracePhase::LoadFc2;
      traceKernel(phase, true, i);
      auto next = task.fc1 ? computeCtx.fc1.load_A(params.fc1, aPipeline, state, sharedStorage.tensors,
                                                  coord, 0, kTiles, weightEvictNormal)
                           : computeCtx.fc2.load_A(params.fc2, aPipeline, state, sharedStorage.tensors,
                                                  coord, 0, kTiles, weightEvictNormal);
      if (hasNext) nextTask = scheduler.taskAt(params, i + numClusters, nTokens);
      traceKernel(phase, false, i);
      state = next;
    }
  }
  aPipeline.producer_tail(state);
}

template <bool Local, class KernelParams, class Storage, class LoadB, class ComputeCtx>
MSCCLPP_DEVICE_INLINE void runW8A16LoadBStage(const KernelParams& params, Storage& sharedStorage, LoadB& bPipeline,
                                              ComputeCtx& computeCtx, const KernelThreadLocation& threadLoc,
                                              int nTokens) {
  if (threadLoc.laneIdx != 0) return;
  const int hidden = params.config.hidden;
  const int intermediate = params.config.intermediate;
  auto state = producerStart<LoadB>();
  computeCtx.fc1.prefetch(params.fc1);
  computeCtx.fc2.prefetch(params.fc2);
  const auto scheduler =
      makeTaskScheduler<KernelParams>(computeCtx.nTokenTiles, computeCtx.nLocalExperts, hidden, intermediate);
  const int numClusters = gridDim.x / KernelParams::Tiles::ClusterSize;
  if (threadLoc.clusterIdx < computeCtx.nTasks) {
    Task nextTask = scheduler.taskAt(params, threadLoc.clusterIdx, nTokens);
    for (int i = threadLoc.clusterIdx; i < computeCtx.nTasks; i += numClusters) {
      Task task = nextTask;
      const bool hasNext = (i + numClusters) < computeCtx.nTasks;
      auto coord = taskCoord(params, task, threadLoc.ctaIdx);
      int kTiles = taskKTiles(params, task, hidden, intermediate);
      auto phase = task.fc1 ? KernelTracePhase::LoadFc1 : KernelTracePhase::LoadFc2;
      traceKernel(phase, true, i);
      loadW8A16ActivationTiles<Local>(params, task, hidden, intermediate, kTiles, [&](int first, int count) {
        state = task.fc1 ? computeCtx.fc1.load_B(params.fc1, bPipeline, state, sharedStorage.tensors,
                                                 coord, first, count, task.tokens.nTokens)
                         : computeCtx.fc2.load_B(params.fc2, bPipeline, state, sharedStorage.tensors,
                                                 coord, first, count, task.tokens.nTokens);
      });
      if (hasNext) nextTask = scheduler.taskAt(params, i + numClusters, nTokens);
      traceKernel(phase, false, i);
    }
  }
  bPipeline.producer_tail(state);
}

template <bool Local, class KernelParams, class Storage, class LoadA, class Transform, class ComputeCtx>
MSCCLPP_DEVICE_INLINE void runW8A16TransformStage(const KernelParams& params, Storage& sharedStorage, LoadA& aPipeline,
                                                  Transform& tPipeline, ComputeCtx& computeCtx,
                                                  const KernelThreadLocation& threadLoc, int nTokens) {
  runW8A16TransformRole<Local>(params, nTokens, sharedStorage, aPipeline, tPipeline, computeCtx.accumulators,
                               params.config.hidden, params.config.intermediate, threadLoc.clusterIdx,
                               computeCtx.nTasks, computeCtx.nTokenTiles, computeCtx.nLocalExperts);
}

template <bool Local, class Schedule, class KernelParams, class Storage, class Pipelines, class ComputeCtx>
MSCCLPP_DEVICE_INLINE void runW8A16MmaStage(const KernelParams& params, Storage& sharedStorage, Pipelines& pipelines,
                                            ComputeCtx& computeCtx, const KernelThreadLocation& threadLoc,
                                            int nTokens) {
  runW8A16MmaRole<Local, Schedule>(
      params, nTokens, sharedStorage, pipelines.loadB, pipelines.transform, pipelines.accumulate,
      computeCtx.accumulators, params.config.hidden, params.config.intermediate, threadLoc.warpIdx, threadLoc.ctaIdx,
      threadLoc.clusterIdx, computeCtx.nTasks, computeCtx.nTokenTiles, computeCtx.nLocalExperts);
}

template <class Schedule, int LocalMode, class KernelParams, class Storage, class Accumulate, class ComputeCtx>
MSCCLPP_DEVICE_INLINE void runW8A16EpilogueStage(const KernelParams& params, Storage& sharedStorage,
                                                 Accumulate& cPipeline, ComputeCtx& computeCtx,
                                                 const KernelThreadLocation& threadLoc, int nTokens,
                                                 __bfloat16* output) {
  runEpilogueRole<Schedule, LocalMode>(params, sharedStorage, cPipeline, computeCtx.accumulators, params.config.hidden,
                                       params.config.intermediate, threadLoc.clusterIdx, computeCtx.nTasks, output,
                                       nTokens);
}

template <bool Local, class Schedule, class RoleFn>
MSCCLPP_DEVICE_INLINE void dispatchW8A16WarpRole(int warpIdx, RoleFn&& runRole) {
  using Types = W8A16CollectiveTypesT<Local>;
  if constexpr (!Local) {
    static_assert(4 * Types::EpilogueRegisters + 4 * Types::TransformRegisters + 4 * Types::DispatchRegisters +
                      4 * TransferRegisters <=
                  Types::NumWarps * EntryRegisters);
  }
  if (warpIdx >= Schedule::EpilogueBegin && warpIdx < Schedule::EpilogueEnd) {
    runRole.template operator()<WarpRole::Epilogue>();
  } else if (warpIdx >= Schedule::TransformBegin && warpIdx < Schedule::TransformEnd) {
    runRole.template operator()<WarpRole::Transform>();
  } else if constexpr (!Local) {
    if (warpIdx >= Schedule::DispatchBegin && warpIdx < Schedule::DispatchEnd) {
      warpgroupRegDealloc<Types::DispatchRegisters>();
      runRole.template operator()<WarpRole::Dispatch>();
      return;
    }
    warpgroupRegDealloc<TransferRegisters>();
    if (warpIdx == Schedule::LoadAWarp) {
      runRole.template operator()<WarpRole::LoadA>();
    } else if (warpIdx == Schedule::LoadBWarp) {
      runRole.template operator()<WarpRole::LoadB>();
    } else if (warpIdx == Schedule::MmaWarp) {
      runRole.template operator()<WarpRole::Mma>();
    }
  } else {
    warpgroupRegDealloc<TransferRegisters>();
    if (warpIdx == Schedule::LoadAWarp) {
      runRole.template operator()<WarpRole::LoadA>();
    } else if (warpIdx == Schedule::LoadBWarp) {
      runRole.template operator()<WarpRole::LoadB>();
    } else if (warpIdx == Schedule::MmaWarp) {
      runRole.template operator()<WarpRole::Mma>();
    }
  }
}

template <int Hidden, bool Local, class KernelParams, class Storage, class Allocator>
MSCCLPP_DEVICE_INLINE void finishW8A16KernelStage(const KernelParams& params, Storage& sharedStorage,
                                                  Allocator& allocator, int warpIdx, int nTokens, __bfloat16* output) {
  (void)output;
  using Schedule = typename W8A16CollectiveTypesT<Local>::WarpSchedule;
  constexpr int RestoreRegisters = Local ? LocalEntryRegisters : EntryRegisters;
  __syncthreads();
  if ((warpIdx >= Schedule::EpilogueBegin && warpIdx < Schedule::EpilogueEnd) ||
      (warpIdx >= Schedule::TransformBegin && warpIdx < Schedule::TransformEnd)) {
    warpgroupRegDealloc<RestoreRegisters>();
  } else {
    warpgroupRegAlloc<RestoreRegisters>();
  }
  clusterSync();
  if (warpIdx == Schedule::MmaWarp) {
    allocator.release_allocation_lock();
    allocator.free(sharedStorage.tmem, 512);
  }
  if constexpr (!Local) {
    if (!params.config.computeOnly) completeStandaloneCombineEpoch(params, nTokens);
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_W8A16_ROLES_CUH_
