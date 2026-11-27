// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Selected SM100-family TMA issue and layouts derived from CUTLASS
// 147295a3d4b75f3aeff247c25b8927cea9a7006a (see ../LICENSE.cutlass).
// Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MSCCLPP_EXT_MEGAMOE_TMA_ISSUE_CUH_
#define MSCCLPP_EXT_MEGAMOE_TMA_ISSUE_CUH_

#include <cuda.h>
#include <cstdint>
#include <type_traits>
#include <mscclpp/device.hpp>

#include "megamoe_kernel.hpp"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

// Preserve the kernel-parameter ABI of the CuTe atom: descriptor at byte 0,
// one 128-byte-aligned slot formerly occupied by the static coordinate trait.
struct alignas(128) MoeTmaSlot {
  CUtensorMap map;
  uint8_t reserved[128];
};
static_assert(sizeof(CUtensorMap) == 128 && sizeof(MoeTmaSlot) == 256 && alignof(MoeTmaSlot) == 128);

enum class MoeTmaCache : uint64_t {
  EVICT_NORMAL = 0x1000000000000000,
  EVICT_FIRST = 0x12F0000000000000,
  EVICT_LAST = 0x14F0000000000000
};

struct MoeTmaTile3 {
  int offsetBytes;
  int32_t c0, c1, c2;
};
struct MoeTmaTile4 {
  int offsetBytes;
  int32_t c0, c1, c2, c3;
};

// Data maps are (K, row, expert). A 128-byte K box occupies consecutive
// per-CTA rows; the box's extent is 64 elements only for BF16.
template <int TileK, int CtaRows, int BoxK = 128>
__host__ __device__ constexpr MoeTmaTile3 dataTmaTile(int stage, int box, int row, int kTile, int expert) {
  static_assert(BoxK == 128 || BoxK == 64);
  return {(stage * (TileK / BoxK) + box) * CtaRows * 128, kTile * TileK + box * BoxK, row, expert};
}

template <int TileK, int ClusterSize, bool Weight>
__host__ __device__ constexpr MoeTmaTile4 w4ScaleTmaTile(int stage, int cta, int tile, int kTile, int expert) {
  constexpr int Blocks = TileK / 128;
  constexpr int StageBytes = 512 * Blocks;
  const int rowBlock = Weight ? tile * ClusterSize + cta : tile;
  // For SFB the scale MMA's one-CTA projection has two slice owners in C2.
  // They write disjoint halves of the stage and multicast both halves.
  const int slice = Weight ? 0 : cta;
  const int offset = stage * StageBytes + slice * (StageBytes / ClusterSize);
  if constexpr (TileK == 128) return {offset, slice * (256 / ClusterSize), rowBlock, kTile, expert};
  else return {offset, 0, kTile * Blocks + slice * (Blocks / ClusterSize), rowBlock, expert};
}

template <int ClusterSize>
__host__ __device__ constexpr int activationTmaRow(int tile, int cta, int validRows) {
  return tile * 128 + cta * (((validRows + 15) / 16) * 16) / ClusterSize;
}

__host__ __device__ constexpr MoeTmaTile3 w8ScaleTmaTile(int stage, int row, int kTile, int expert) {
  return {stage * 512, row, kTile * 4, expert};
}

MSCCLPP_DEVICE_INLINE void prefetchMoeTma(const CUtensorMap* desc) {
  asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(desc)) : "memory");
}

template <int Group, bool Multicast = false, class Tile>
MSCCLPP_DEVICE_INLINE void issueMoeTma(const CUtensorMap* desc, const uint64_t* barrier, void* smem, Tile tile,
                                       MoeTmaCache hint, uint16_t mask = 0) {
  static_assert(Group == 1 || Group == 2);
  const uint64_t map = reinterpret_cast<uint64_t>(desc);
  const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(static_cast<uint8_t*>(smem) + tile.offsetBytes));
  // In a two-CTA issue, either CTA updates the leader CTA0's transaction bytes.
  const uint32_t mbar = static_cast<uint32_t>(__cvta_generic_to_shared(barrier)) &
                       (Group == 2 ? 0xfeffffffu : 0xffffffffu);
  const uint64_t cache = static_cast<uint64_t>(hint);
  const int32_t c0 = tile.c0, c1 = tile.c1, c2 = tile.c2;
  if constexpr (std::is_same_v<Tile, MoeTmaTile3>) {
    static_assert(!Multicast);
    if constexpr (Group == 2) {
      asm volatile("cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                   " [%0], [%1, {%3, %4, %5}], [%2], %6;"
                   :: "r"(dst), "l"(map), "r"(mbar), "r"(c0), "r"(c1), "r"(c2), "l"(cache) : "memory");
    } else {
      asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                   " [%0], [%1, {%3, %4, %5}], [%2], %6;"
                   :: "r"(dst), "l"(map), "r"(mbar), "r"(c0), "r"(c1), "r"(c2), "l"(cache) : "memory");
    }
  } else {
    static_assert(std::is_same_v<Tile, MoeTmaTile4> && (!Multicast || Group == 2));
    const int32_t c3 = tile.c3;
    if constexpr (Multicast) {
      asm volatile("cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
                   " [%0], [%1, {%4, %5, %6, %7}], [%2], %3, %8;"
                   :: "r"(dst), "l"(map), "r"(mbar), "h"(mask), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "l"(cache) : "memory");
    } else if constexpr (Group == 2) {
      asm volatile("cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                   " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
                   :: "r"(dst), "l"(map), "r"(mbar), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "l"(cache) : "memory");
    } else {
      asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                   " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
                   :: "r"(dst), "l"(map), "r"(mbar), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "l"(cache) : "memory");
    }
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_TMA_ISSUE_CUH_
