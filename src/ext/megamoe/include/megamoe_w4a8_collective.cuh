// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Selected SM100 block-scaled layouts and descriptor fields from CUTLASS
// 147295a3d4b75f3aeff247c25b8927cea9a7006a. See ../LICENSE.cutlass.
// Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MSCCLPP_EXT_MEGAMOE_W4A8_COLLECTIVE_CUH_
#define MSCCLPP_EXT_MEGAMOE_W4A8_COLLECTIVE_CUH_

#include <cstddef>

#include "megamoe_activation_tma.hpp"
#include "megamoe_primitives.cuh"
#include "megamoe_tma_issue.cuh"
#include "megamoe_trace.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

// Byte-compatible with the selected blocked-product carrier. A 128x128
// block has four K32 scales per row; K blocks precede row blocks.
struct W4ScaleLayout {
  int rows, kBlocks, experts, rowStride, expertStride;

  static constexpr W4ScaleLayout make(int rows, int k, int experts, bool broadcast) {
    const int rowBlocks = (rows + 127) / 128;
    const int kBlocks = (k + 127) / 128;
    const int rowStride = kBlocks * 512;
    return {rowBlocks, kBlocks, experts, rowStride, broadcast ? 0 : rowBlocks * rowStride};
  }

  __host__ __device__ constexpr int operator()(int row, int k, int expert) const {
    return expert * expertStride + (row / 128) * rowStride + (k / 128) * 512 +
           (row % 32) * 16 + ((row / 32) % 4) * 4 + (k / 32) % 4;
  }

  constexpr size_t bytes() const { return size_t(experts - 1) * expertStride + size_t(rows) * rowStride; }
};
static_assert(sizeof(W4ScaleLayout) == 20 && alignof(W4ScaleLayout) == 4);

// Selected SM100f Major-K descriptors: version 1 and leading offset 1.
// SW128 data stride is 64, unswizzled scale stride is 8 (units of 16 bytes).
__host__ __device__ constexpr uint64_t w4SmemDescriptor(uint32_t base, int offset, bool scale) {
  const uint64_t format = (uint64_t{1} << 46) | (uint64_t{1} << 16) |
                          (scale ? (uint64_t{8} << 32) : ((uint64_t{2} << 61) | (uint64_t{64} << 32)));
  return (format | ((base >> 4) & 0x3fffu)) + offset / 16;
}

template <int TileK, int CtaRows>
__host__ __device__ constexpr uint64_t w4MmaData(uint32_t base, int stage, int k32) {
  return w4SmemDescriptor(base, (stage * (TileK / 128) + k32 / 4) * CtaRows * 128 + k32 % 4 * 32, false);
}

template <int TileK>
__host__ __device__ constexpr uint64_t w4UtccpSource(uint32_t base, int stage, int k128) {
  return w4SmemDescriptor(base, (stage * (TileK / 128) + k128) * 512, true);
}

__host__ __device__ constexpr uint32_t w4MmaScale(uint32_t base, int k32) {
  return base + (k32 / 4) * 4 + (uint32_t(k32 % 4) << 30);
}

template <int C>
__host__ __device__ constexpr uint32_t w4Instruction(int rows, uint32_t scaleA, uint32_t scaleB) {
  // M/16, E8M0, N/8, E2M1 A, E4M3 B, and the two scale-byte selectors.
  return (uint32_t(8 * C) << 24) | (1u << 23) | (uint32_t((rows + 15) / 16 * 2) << 17) | (5u << 7) |
         ((scaleA >> 30) << 29) | ((scaleB >> 30) << 4);
}

template <int TileN, int TileK, int LoadStages, int ClusterSize>
struct W4A8Mainloop {
  static_assert(TileN == 128 && (ClusterSize == 1 || ClusterSize == 2));
  static constexpr int WeightStageBytes = 128 * TileK; // FP4 expands to one byte per value in SMEM.
  static constexpr int ActivationStageBytes = WeightStageBytes / ClusterSize;
  static constexpr int ScaleStageBytes = 512 * (TileK / 128);
  using LayoutSFA = W4ScaleLayout;
  using LayoutSFB = W4ScaleLayout;
  using MainloopPipeline = MoeTmaUmmaPipe<LoadStages, ClusterSize>;
  using MainloopPipelineState = typename MainloopPipeline::PipelineState;
  using CacheHint = MoeTmaCache;

  struct Arguments {
    const void* ptr_A = nullptr;
    const void* ptr_B = nullptr;
    const void* ptr_SFA = nullptr;
    W4ScaleLayout layout_SFA{};
    const void* ptr_SFB = nullptr;
    W4ScaleLayout layout_SFB{};
  };

  struct alignas(128) TensorStorage {
    uint8_t smem_A[WeightStageBytes * LoadStages];
    uint8_t smem_B[ActivationStageBytes * LoadStages];
    uint8_t smem_SFA[ScaleStageBytes * LoadStages];
    uint8_t smem_SFB[ScaleStageBytes * LoadStages];
  };

  struct MmaParams {
    uint32_t dataA, dataB, sharedScaleA, sharedScaleB, scaleA, scaleB;
  };

  static constexpr uint32_t WeightTransactionBytes = ClusterSize * (WeightStageBytes / 2 + ScaleStageBytes);
  static constexpr uint32_t SfbTransactionBytes = ClusterSize * ScaleStageBytes;
  static constexpr uint32_t ActivationTransactionBytes = TileN * TileK + SfbTransactionBytes;
  static constexpr uint32_t TmaTransactionBytes = WeightTransactionBytes + ActivationTransactionBytes;

  struct Params {
    MoeTmaSlot tma_load_a, tma_load_b, tma_load_sfa, tma_load_sfb;
    MoeTmaSlot reservedA, reservedB, reservedSFA, reservedSFB;
    W4ScaleLayout layout_SFA, layout_SFB;
    dim3 reservedCluster{0, 0, 0};
    void* reservedFormatA = nullptr;
    void* reservedFormatB = nullptr;
    CUtensorMap tma_desc_b_64, tma_desc_b_32;
  };
  static_assert(sizeof(Params) == 2432 && alignof(Params) == 128);
  static_assert(offsetof(Params, layout_SFA) == 2048 && offsetof(Params, layout_SFB) == 2068 &&
                offsetof(Params, reservedCluster) == 2088 && offsetof(Params, tma_desc_b_32) == 2304);

  MSCCLPP_DEVICE_INLINE W4A8Mainloop(const Params& params, uint32_t ctaRank) : ctaRank_(ctaRank), params_(params) {}

  static bool can_implement(const ProblemShape& shape, const Arguments& args) {
    auto [m, n, k, experts] = shape;
    auto samePlane = [](W4ScaleLayout a, W4ScaleLayout b) {
      return a.rows == b.rows && a.kBlocks == b.kBlocks && a.rowStride == b.rowStride;
    };
    return k % 128 == 0 && samePlane(W4ScaleLayout::make(m, k, experts, false), args.layout_SFA) &&
           samePlane(W4ScaleLayout::make(n, k, experts, true), args.layout_SFB);
  }

  // Three 128-column FP32 accumulator slots precede both scale fragments.
  MSCCLPP_DEVICE_INLINE static MmaParams mma_init(uint32_t accumulators, TensorStorage& shared) {
    const uint32_t scaleA = accumulatorStage(accumulators, 3);
    return {uint32_t(__cvta_generic_to_shared(shared.smem_A)), uint32_t(__cvta_generic_to_shared(shared.smem_B)),
            uint32_t(__cvta_generic_to_shared(shared.smem_SFA)), uint32_t(__cvta_generic_to_shared(shared.smem_SFB)),
            scaleA, scaleA + TileK / 32};
  }

  MSCCLPP_DEVICE_INLINE static void load_tail(MainloopPipeline pipeline, MainloopPipelineState state) {
    pipeline.producer_tail(state);
  }

  static Params to_underlying_arguments(const ProblemShape& shape, const Arguments& args, void*) {
    auto [m, n, k, experts] = shape;
    Params params{};
    params.tma_load_a.map = makeDataMap<4, true, 128>(args.ptr_A, m, k, experts);
    params.tma_load_b.map = makeActivationSubtileMap<1, ClusterSize, 128>(args.ptr_B, n, k, experts);
    params.tma_load_sfa.map = makeW4ScaleMap<TileK, ClusterSize, true>(args.ptr_SFA, m, k, experts);
    params.tma_load_sfb.map = makeW4ScaleMap<TileK, ClusterSize, false>(args.ptr_SFB, n, k, experts);
    params.layout_SFA = args.layout_SFA;
    params.layout_SFB = args.layout_SFB;
    params.tma_desc_b_64 = makeActivationSubtileMap<1, ClusterSize, 64>(args.ptr_B, n, k, experts);
    params.tma_desc_b_32 = makeActivationSubtileMap<1, ClusterSize, 32>(args.ptr_B, n, k, experts);
    return params;
  }

  MSCCLPP_DEVICE_INLINE void prefetch() const {
    if (electMoeLane()) {
      prefetchMoeTma(&params_.tma_load_a.map);
      prefetchMoeTma(&params_.tma_load_b.map);
      prefetchMoeTma(&params_.tma_load_sfa.map);
      prefetchMoeTma(&params_.tma_load_sfb.map);
      prefetchMoeTma(&params_.tma_desc_b_64);
      prefetchMoeTma(&params_.tma_desc_b_32);
    }
  }

  template <int Operands = 0, ActivationDtype ActDtype = ActivationDtype::Mxfp8>
  MSCCLPP_DEVICE_INLINE MainloopPipelineState load(MainloopPipeline pipe, MainloopPipelineState state,
                                                  TensorStorage& shared, MoeTaskCoord coord, int kFirst, int kTiles,
                                                  int validRows, bool weightEvictNormal) const {
    const int m = coord.m / ClusterSize;
    const int n = coord.n;
    const int expert = coord.expert;
    const int ctaIdx = int(ctaRank_);
    const int weightRow = (m * ClusterSize + ctaIdx) * 128;
    const int activationRow = activationTmaRow<ClusterSize>(n, ctaIdx, validRows);
    const auto weightHint = weightEvictNormal ? CacheHint::EVICT_NORMAL : CacheHint::EVICT_FIRST;
    const CUtensorMap* bDesc = &params_.tma_load_b.map;
    uint32_t actTxBytes = ActivationTransactionBytes;
    if (validRows <= 32) {
      bDesc = &params_.tma_desc_b_32;
      actTxBytes = 32 * TileK + SfbTransactionBytes;
    } else if (validRows <= 64) {
      bDesc = &params_.tma_desc_b_64;
      actTxBytes = 64 * TileK + SfbTransactionBytes;
    }
    if constexpr (ActDtype == ActivationDtype::Fp8Static) actTxBytes -= SfbTransactionBytes;
    const uint32_t txBytes =
        Operands == 1 ? WeightTransactionBytes : (Operands == 2 ? actTxBytes : WeightTransactionBytes + actTxBytes);

    uint64_t waitNs = 0, issueNs = 0;
    const int iterations = kTiles;
    for (; kTiles > 0; --kTiles, ++kFirst) {
      auto start = traceKernelClock();
      pipe.producer_acquire(state);
      waitNs += traceKernelClock() - start;
      auto* barrier = pipe.producer_get_barrier(state);
      const int stage = state.index();
      start = traceKernelClock();
      if (electMoeLane()) {
        if (ctaIdx == 0) arriveMoeTmaExpect(barrier, txBytes);
        if constexpr (Operands != 2) {
#pragma unroll
          for (int box = 0; box < TileK / 128; ++box)
            issueMoeTma<ClusterSize>(&params_.tma_load_a.map, barrier, shared.smem_A,
                                     dataTmaTile<TileK, 128>(stage, box, weightRow, kFirst, expert), weightHint);
          issueMoeTma<ClusterSize>(&params_.tma_load_sfa.map, barrier, shared.smem_SFA,
                                  w4ScaleTmaTile<TileK, ClusterSize, true>(stage, ctaIdx, m, kFirst, expert), weightHint);
        }
        if constexpr (Operands != 1) {
#pragma unroll
          for (int box = 0; box < TileK / 128; ++box)
            issueMoeTma<ClusterSize>(bDesc, barrier, shared.smem_B,
                                     dataTmaTile<TileK, 128 / ClusterSize>(stage, box, activationRow, kFirst, expert),
                                     CacheHint::EVICT_LAST);
          if constexpr (ActDtype == ActivationDtype::Mxfp8)
            issueMoeTma<ClusterSize, ClusterSize == 2>(
                &params_.tma_load_sfb.map, barrier, shared.smem_SFB,
                w4ScaleTmaTile<TileK, ClusterSize, false>(stage, ctaIdx, n, kFirst, expert),
                CacheHint::EVICT_LAST, (1u << ClusterSize) - 1u);
        }
      }
      issueNs += traceKernelClock() - start;
      ++state;
    }
    traceKernelTotal(KernelTracePhase::LoadAcquire, waitNs, iterations);
    traceKernelTotal(KernelTracePhase::TmaIssue, issueNs, iterations);
    return state;
  }

 private:
  uint32_t ctaRank_;
  const Params& params_;
};

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_W4A8_COLLECTIVE_CUH_
