// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_DETAIL_PORT_CHANNEL_ASSERT_HPP_
#define MSCCLPP_DETAIL_PORT_CHANNEL_ASSERT_HPP_

#include <cstdio>
#include <cstdlib>

#include "../device.hpp"

namespace mscclpp::detail {

// Unlike debug assertions, unsupported channel operations must fail in release kernels too.
[[noreturn]] MSCCLPP_HOST_DEVICE_INLINE void portChannelFailure(const char* message) {
#if defined(__CUDA_ARCH__)
  printf("MSCCL++ PortChannel: %s\n", message);
  asm volatile("trap;" ::: "memory");
  for (;;) {
  }  // The trap never returns; retain that contract for the compiler too.
#elif defined(__HIP_DEVICE_COMPILE__)
  printf("MSCCL++ PortChannel: %s\n", message);
  __builtin_trap();
#else
  std::fprintf(stderr, "MSCCL++ PortChannel: %s\n", message);
  std::abort();
#endif
}

MSCCLPP_HOST_DEVICE_INLINE void portChannelRequire(bool condition, const char* message) {
  if (!condition) portChannelFailure(message);
}

}  // namespace mscclpp::detail

#endif  // MSCCLPP_DETAIL_PORT_CHANNEL_ASSERT_HPP_
