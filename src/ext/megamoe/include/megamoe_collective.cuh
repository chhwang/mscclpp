// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_COLLECTIVE_CUH_
#define MSCCLPP_EXT_MEGAMOE_COLLECTIVE_CUH_

#include <mscclpp/device.hpp>

#include "megamoe_accumulator_pipe.cuh"
#include "megamoe_kernel.hpp"
#include "megamoe_load_pipes.cuh"
#include "megamoe_specialization.hpp"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {
struct ProblemShape { int m, n, k, experts; };
struct MoeTaskCoord { int m, n, expert; };
}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#include "megamoe_w4a8_collective.cuh"
#include "megamoe_w8a16_collective.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <bool Local = false>
struct W8A16CollectiveTypesT {
  static constexpr int NumWarps = Local ? 12 : 16;
  static constexpr int EpilogueWarps = 4;
  static constexpr int EpilogueRegisters = Local ? 232 : 224;
  static constexpr int TransformRegisters = Local ? 232 : 176;
  static constexpr int DispatchRegisters = 80;
  static constexpr int TransferRegisters = 32;
  static constexpr int LoadWarps = 2;
  static constexpr bool SplitPipelines = false;
  static constexpr int EpilogueTokens = 32;
  static constexpr int DispatchChunk = W4DispatchChunk;
  static constexpr int DispatchWarps = 4;
  static constexpr int DispatchStages = W8DispatchStages;
  static constexpr bool WeightMxfp4 = false;
  static constexpr bool LocalExpert = Local;
  using WarpSchedule = W8A16WarpSchedule<Local>;
  using Tiles = TilePolicy<128, 128>;
  using Mainloop = W8A16Mainloop;
  using LoadA = typename Mainloop::Load2TransformPipeline;
  using LoadB = typename Mainloop::Load2MmaPipeline;
  using Transform = typename Mainloop::Transform2MmaPipeline;
  using Accumulate = typename Mainloop::Mma2AccumPipeline;
};

template <int TileN_, int TileK_, int LoadStages_, int NumWarps_, int TransferRegisters_, int LoadWarps_,
          bool SplitPipelines_, int EpilogueTokens_, int EpilogueWarps_, int EpilogueRegisters_, int DispatchChunk_,
          int DispatchWarps_, int DispatchStages_>
struct W4A8CollectiveTypesT {
  static constexpr int NumWarps = NumWarps_;
  static constexpr int EpilogueWarps = EpilogueWarps_;
  static constexpr int EpilogueRegisters = EpilogueRegisters_;
  static constexpr int TransferRegisters = TransferRegisters_;
  static constexpr int LoadWarps = LoadWarps_;
  static constexpr bool SplitPipelines = SplitPipelines_;
  static constexpr int EpilogueTokens = EpilogueTokens_;
  static constexpr int DispatchChunk = DispatchChunk_;
  static constexpr int DispatchWarps = DispatchWarps_;
  static constexpr int DispatchStages = DispatchStages_;
  static constexpr bool WeightMxfp4 = true;
  static constexpr bool LocalExpert = false;
  using WarpSchedule = W4A8WarpScheduleT<NumWarps, EpilogueWarps, DispatchWarps>;
  using Tiles = TilePolicy<TileN_, TileK_>;
  using Mainloop = W4A8Mainloop<TileN_, TileK_, LoadStages_, W4ClusterSize>;
  using Load = typename Mainloop::MainloopPipeline;
  using Accumulate = MoeAccumulatorPipeline<3, W4ClusterSize>;
  static_assert(NumWarps == 12 || NumWarps == 16);
  static_assert(LoadWarps == 1 || LoadWarps == 2);
  static_assert(!SplitPipelines || LoadWarps == 2);
  static_assert(EpilogueTokens == 16 || EpilogueTokens == 32);
  static_assert(TileN_ % EpilogueTokens == 0);
  static_assert(EpilogueWarps == 4 || EpilogueWarps == 8);
  static_assert(EpilogueRegisters >= 128 && EpilogueRegisters <= 224 && EpilogueRegisters % 8 == 0);
  static_assert(TransferRegisters >= 32 && TransferRegisters <= 128 && TransferRegisters % 32 == 0);
  static_assert(DispatchChunk % TileK_ == 0);
  static_assert(DispatchWarps == 2 || DispatchWarps == 4 || DispatchWarps == 8);
  static_assert(DispatchStages == 1 || DispatchStages == 2);
  static_assert(WarpSchedule::DispatchEnd <= WarpSchedule::NumWarps);
};

using W4A8CollectiveTypes =
    W4A8CollectiveTypesT<W4TileN, W4TileK, W4LoadStages, W4NumWarps, W4TransferRegisters, W4LoadWarps, W4SplitPipelines,
                         W4EpilogueTokens, W4EpilogueWarps, W4EpilogueRegisters, W4DispatchChunk, W4DispatchWarps,
                         W4DispatchStages>;

static_assert(W4A8CollectiveTypes::Load::Stages == W4LoadStages);

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_COLLECTIVE_CUH_
