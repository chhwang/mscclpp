// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Selected descriptor construction adapted from NVIDIA CUTLASS at
// 147295a3d4b75f3aeff247c25b8927cea9a7006a (see ../LICENSE.cutlass):
// cute/atom/{copy_traits_sm100_tma,copy_traits_sm90_tma}.hpp,
// cute/arch/copy_sm90_desc.hpp and cutlass/cuda_host_adapter.hpp.
// Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MSCCLPP_EXT_MEGAMOE_ACTIVATION_TMA_HPP_
#define MSCCLPP_EXT_MEGAMOE_ACTIVATION_TMA_HPP_

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdint>
#include <cstdio>

#include "megamoe_specialization.hpp"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <unsigned Rank>
CUtensorMap encodeMegaMoeMap(const void* input, CUtensorMapDataType format, const cuuint64_t (&extent)[Rank],
                             const cuuint64_t (&strideBytes)[Rank - 1], const cuuint32_t (&box)[Rank],
                             CUtensorMapSwizzle swizzle, cuuint64_t planeBytes, int contiguousPlanes) {
  static_assert(Rank == 3 || Rank == 4);
  assert((reinterpret_cast<uintptr_t>(input) & 15) == 0);
  for (auto d : extent) assert(d > 0 && d <= (1ull << 32));
  for (auto d : strideBytes) assert(d % 16 == 0 && d < (1ull << 40));
  constexpr cuuint32_t steps[] = {1, 1, 1, 1};
  CUtensorMap map{};
  CUresult result = CUDA_ERROR_UNKNOWN;
#if defined(CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL)
  auto encode = &cuTensorMapEncodeTiled;
#else
  void* entry = nullptr;
  cudaDriverEntryPointQueryResult status;
  const auto error =
      cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &entry, 12000, cudaEnableDefault, &status);
  // CUTLASS can include cudaTypedefs.h inside its namespace. Use the SDK declaration directly.
  auto encode = error == cudaSuccess && status == cudaDriverEntryPointSuccess
                    ? reinterpret_cast<decltype(&::cuTensorMapEncodeTiled)>(entry)
                    : nullptr;
#endif
  if (encode) {
    result = encode(&map, format, Rank, const_cast<void*>(input), extent, strideBytes, box, steps,
                    CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  }
  // Preserve CuTe's short-contiguous-prefix workaround.
  constexpr cuuint64_t PrefixBytes = 128 * 1024;
  if (planeBytes < PrefixBytes && planeBytes * contiguousPlanes < PrefixBytes) map.opaque[1] &= ~(1ull << 21);
  if (result != CUDA_SUCCESS) {
    std::fprintf(stderr, "MegaMoE tensor map: rank=%u format=%d box=%ux%u\n", Rank, int(format), box[0], box[1]);
    std::fprintf(stderr, "Error: Failed to initialize the TMA descriptor %d\n", int(result));
    assert(false);
  }
  return map;
}

// A/B use K-major rows. Only weights advance across experts.
template <int ElementBits, bool Weight, int CtaRows>
CUtensorMap makeDataMap(const void* input, int rows, int width, int experts) {
  static_assert(ElementBits == 4 || ElementBits == 8 || ElementBits == 16);
  constexpr auto format = ElementBits == 4   ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
                          : ElementBits == 8 ? CU_TENSOR_MAP_DATA_TYPE_UINT8
                                             : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  const cuuint64_t extent[] = {cuuint64_t(width), cuuint64_t(rows), cuuint64_t(experts)};
  const cuuint64_t rowBytes = cuuint64_t(width) * ElementBits / 8;
  const cuuint64_t planeBytes = rowBytes * rows;
  const cuuint64_t stride[] = {rowBytes, Weight ? planeBytes : 0};
  // W4 expands each FP4 nibble to one byte in SMEM. Its box is 128 elements like E4M3.
  constexpr cuuint32_t box[] = {ElementBits == 16 ? 64u : 128u, CtaRows, 1};
  return encodeMegaMoeMap(input, format, extent, stride, box, CU_TENSOR_MAP_SWIZZLE_128B, planeBytes,
                          Weight ? experts : 1);
}

template <int ElementBytes, int ClusterSize, int SubtileRows>
CUtensorMap makeActivationSubtileMap(const void* input, int rows, int width, int experts) {
  static_assert(ElementBytes == 1 || ElementBytes == 2);
  static_assert(ClusterSize == 1 || ClusterSize == 2);
  static_assert(SubtileRows == 32 || SubtileRows == 64 || SubtileRows == 128);
  return makeDataMap<ElementBytes * 8, false, SubtileRows / ClusterSize>(input, rows, width, experts);
}

// W4 stores 512 bytes for each 128-row by 128-K block, K blocks first. The
// descriptor groups these as UINT16 pairs; the first box spans at most 256 pairs.
template <int TileK, int ClusterSize, bool Weight>
CUtensorMap makeW4ScaleMap(const void* input, int rows, int width, int experts) {
  assert(rows % 128 == 0 && width % 128 == 0);
  constexpr int blocksPerCta = TileK / 128 / (Weight ? 1 : ClusterSize);
  const cuuint64_t rowBlocks = rows / 128;
  const cuuint64_t kBlocks = width / 128;
  const cuuint64_t planeBytes = rowBlocks * kBlocks * 512;
  constexpr bool KFirst = TileK > 128;
  const cuuint64_t extent[] = {256, KFirst ? kBlocks : rowBlocks, KFirst ? rowBlocks : kBlocks, cuuint64_t(experts)};
  const cuuint64_t stride[] = {KFirst ? 512 : kBlocks * 512, KFirst ? kBlocks * 512 : 512, Weight ? planeBytes : 0};
  constexpr cuuint32_t box[] = {TileK == 128 && !Weight ? 256u / ClusterSize : 256u,
                                blocksPerCta > 0 ? blocksPerCta : 1u, 1, 1};
  return encodeMegaMoeMap(input, CU_TENSOR_MAP_DATA_TYPE_UINT16, extent, stride, box, CU_TENSOR_MAP_SWIZZLE_NONE,
                          planeBytes, Weight ? experts : 1);
}

inline CUtensorMap makeW8ScaleMap(const void* input, int rows, int width, int experts) {
  assert(width % 32 == 0);
  const cuuint64_t extent[] = {cuuint64_t(rows), cuuint64_t(width / 32), cuuint64_t(experts)};
  const cuuint64_t planeBytes = extent[0] * extent[1];
  const cuuint64_t stride[] = {extent[0], planeBytes};
  constexpr cuuint32_t box[] = {128, 4, 1};
  return encodeMegaMoeMap(input, CU_TENSOR_MAP_DATA_TYPE_UINT8, extent, stride, box, CU_TENSOR_MAP_SWIZZLE_NONE,
                          planeBytes, experts);
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_ACTIVATION_TMA_HPP_
