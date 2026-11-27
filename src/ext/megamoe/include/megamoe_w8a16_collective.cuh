// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Selected SM100 mixed-input layout and descriptor fields from CUTLASS
// 147295a3d4b75f3aeff247c25b8927cea9a7006a. See ../LICENSE.cutlass.
// Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MSCCLPP_EXT_MEGAMOE_W8A16_COLLECTIVE_CUH_
#define MSCCLPP_EXT_MEGAMOE_W8A16_COLLECTIVE_CUH_

#include <cstddef>

#include "megamoe_activation_tma.hpp"
#include "megamoe_primitives.cuh"
#include "megamoe_tma_issue.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

__host__ __device__ constexpr int w8RawOffset(int stage, int lane, int k) {
  // SW128: the low three row bits select the 16-byte K swizzle.
  return stage * 16384 + lane * 128 + (k ^ ((lane & 7) * 16));
}
__host__ __device__ constexpr int w8ScaleOffset(int stage, int lane, int k32) {
  return stage * 512 + k32 * 128 + lane;
}
__host__ __device__ constexpr uint32_t w8WeightAddress(uint32_t base, int stage, int k16, int lane = 0) {
  return base + stage * 64 + k16 * 8 + ((uint32_t(lane) & 96u) << 16);
}
__host__ __device__ constexpr uint64_t w8ActivationDescriptor(uint32_t base, int stage, int k16) {
  return w4SmemDescriptor(base, stage * 16384 + (k16 / 4) * 8192 + (k16 % 4) * 32, false);
}
__host__ __device__ constexpr uint32_t w8Instruction(int rows) {
  // M=256, K-major BF16 A and B, FP32 C; N is rounded up to 16.
  return (16u << 24) | (uint32_t((rows + 15) / 16 * 2) << 17) | (1u << 10) | (1u << 7) | (1u << 4);
}

struct W8A16Mainloop {
  struct Arguments {
    const void* ptr_A = nullptr;
    const void* ptr_B = nullptr;
    const void* ptr_S = nullptr;
  };

  struct alignas(1024) TensorStorage {
    struct {
      alignas(512) uint8_t smem_A[16384 * W8LoadStages];
      alignas(1024) uint8_t smem_B[16384 * W8LoadStages];
      uint8_t smem_scale[512 * W8LoadStages];
      uint8_t reservedZero;
    } input;
    uint16_t reservedCompute;
  };
  static_assert(sizeof(TensorStorage) == 167936 && alignof(TensorStorage) == 1024);
  static_assert(offsetof(TensorStorage, input.smem_B) == 81920 && offsetof(TensorStorage, input.smem_scale) == 163840 &&
                offsetof(TensorStorage, reservedCompute) == 166912);

  using Load2TransformPipeline = MoeW8WeightPipe<W8LoadStages>;
  using Load2MmaPipeline = MoeTmaUmmaPipe<W8LoadStages, 2>;
  using Transform2MmaPipeline = MoeW8TransformPipe<W8TransformStages>;
  using Mma2AccumPipeline = MoeAccumulatorPipeline<W8AccumStages, 2>;
  using Load2MmaPipelineState = typename Load2MmaPipeline::PipelineState;
  using Load2TransformPipelineState = typename Load2TransformPipeline::PipelineState;
  static constexpr uint32_t TmaTransactionBytes_A = 128 * 128 + 128 * 4;
  static constexpr uint32_t TmaTransactionBytes_B = 128 * 128 * 2;

  struct Params {
    MoeTmaSlot tma_load_scale, reservedZero, tma_load_a, tma_load_b, reservedA, reservedB;
    dim3 reservedCluster{0, 0, 0};
    uint32_t reservedTransactionBytes = TmaTransactionBytes_A + TmaTransactionBytes_B;
    // Byte-compatible with the two dynamic strides of the old K-major carrier.
    int64_t dA[2]{}, dB[2]{};
    CUtensorMap tma_desc_b_64, tma_desc_b_32;
  };
  static_assert(sizeof(Params) == 1920 && alignof(Params) == 128);
  static_assert(offsetof(Params, reservedCluster) == 1536 && offsetof(Params, dA) == 1552 &&
                offsetof(Params, dB) == 1568 && offsetof(Params, tma_desc_b_32) == 1792);

  explicit __device__ W8A16Mainloop(uint32_t ctaRank) : ctaRank_(ctaRank) {}

  static bool can_implement(const ProblemShape& shape, const Arguments& args) {
    // K-major A/B and row-major scales, 16-byte vector alignment.
    return shape.k % 16 == 0 && shape.m % 16 == 0 && args.ptr_S;
  }

  static Params to_underlying_arguments(const ProblemShape& shape, const Arguments& args, void*) {
    auto [m, n, k, experts] = shape;
    Params params{};
    params.tma_load_a.map = makeDataMap<8, true, 128>(args.ptr_A, m, k, experts);
    params.tma_load_b.map = makeActivationSubtileMap<2, 2, 128>(args.ptr_B, n, k, experts);
    params.tma_load_scale.map = makeW8ScaleMap(args.ptr_S, m, k, experts);
    params.dA[0] = k;
    params.dA[1] = int64_t(m) * k;
    params.dB[0] = k;
    params.tma_desc_b_64 = makeActivationSubtileMap<2, 2, 64>(args.ptr_B, n, k, experts);
    params.tma_desc_b_32 = makeActivationSubtileMap<2, 2, 32>(args.ptr_B, n, k, experts);
    return params;
  }

  MSCCLPP_DEVICE_INLINE static void prefetch(const Params& params) {
    if (threadIdx.x % WARP_SIZE == 0) {
      prefetchMoeTma(&params.tma_load_a.map);
      prefetchMoeTma(&params.tma_load_b.map);
      prefetchMoeTma(&params.tma_load_scale.map);
      prefetchMoeTma(&params.tma_desc_b_64);
      prefetchMoeTma(&params.tma_desc_b_32);
    }
  }

  __host__ __device__ static constexpr uint32_t w8WeightTmemBase(uint32_t accumulator) {
    return accumulatorStage(accumulator, W8AccumStages);
  }

  MSCCLPP_DEVICE_INLINE Load2TransformPipelineState load_A(const Params& params, Load2TransformPipeline pipe,
                                                            Load2TransformPipelineState state, TensorStorage& shared,
                                                            MoeTaskCoord coord, int kFirst, int kTiles,
                                                            bool weightEvictNormal) const {
    const int row = (coord.m / 2 * 2 + int(ctaRank_)) * 128;
    const auto hint = weightEvictNormal ? MoeTmaCache::EVICT_NORMAL : MoeTmaCache::EVICT_FIRST;
    auto ready = pipe.producer_try_acquire(state, kTiles <= 0);
#pragma unroll 1
    for (; kTiles > 0; --kTiles, ++kFirst) {
      pipe.producer_acquire(state, ready);
      const int stage = state.index();
      auto* barrier = pipe.producer_get_barrier(state);
      ++state;
      ready = pipe.producer_try_acquire(state, kTiles <= 1);
      issueMoeTma<1>(&params.tma_load_a.map, barrier, shared.input.smem_A,
                     dataTmaTile<128, 128>(stage, 0, row, kFirst, coord.expert), hint);
      issueMoeTma<1>(&params.tma_load_scale.map, barrier, shared.input.smem_scale,
                     w8ScaleTmaTile(stage, row, kFirst, coord.expert), hint);
    }
    return state;
  }

  MSCCLPP_DEVICE_INLINE Load2MmaPipelineState load_B(const Params& params, Load2MmaPipeline pipe,
                                                      Load2MmaPipelineState state, TensorStorage& shared,
                                                      MoeTaskCoord coord, int kFirst, int kTiles, int validRows) const {
    const int ctaIdx = int(ctaRank_);
    const int row = activationTmaRow<2>(coord.n, ctaIdx, validRows);
    const CUtensorMap* bDesc = &params.tma_load_b.map;
    uint32_t actTxBytes = TmaTransactionBytes_B;
    if (validRows <= 32) {
      bDesc = &params.tma_desc_b_32;
      actTxBytes = 32 * 128 * 2;
    } else if (validRows <= 64) {
      bDesc = &params.tma_desc_b_64;
      actTxBytes = 64 * 128 * 2;
    }
    auto ready = pipe.producer_try_acquire(state, kTiles <= 0);
#pragma unroll 1
    for (; kTiles > 0; --kTiles, ++kFirst) {
      pipe.producer_acquire(state, ready);
      const int stage = state.index();
      auto* barrier = pipe.producer_get_barrier(state);
      if (ctaIdx == 0) arriveMoeTmaExpect(barrier, actTxBytes);
      ++state;
      ready = pipe.producer_try_acquire(state, kTiles <= 1);
#pragma unroll
      for (int box = 0; box < 2; ++box)
        issueMoeTma<2>(bDesc, barrier, shared.input.smem_B,
                       dataTmaTile<128, 64, 64>(stage, box, row, kFirst, coord.expert), MoeTmaCache::EVICT_NORMAL);
    }
    return state;
  }

 private:
  uint32_t ctaRank_;
};

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_W8A16_COLLECTIVE_CUH_
