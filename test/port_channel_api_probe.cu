// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Also compile directly against an installed prefix, with no USE_IBVERBS/GPUNetIO definitions.
#include <mscclpp/port_channel.hpp>
#include <type_traits>

static_assert(std::is_trivially_copyable_v<mscclpp::BasePortChannelDeviceHandle>);
static_assert(std::is_trivially_copyable_v<mscclpp::PortChannelDeviceHandle>);
constexpr mscclpp::PortChannelDeviceHandle defaultPortHandle;
__constant__ mscclpp::PortChannelDeviceHandle constantPortHandle;

template <class T>
concept ExposesBackend = requires(T t) { t.device_; } || requires(T t) { t.state_; };
template <class T>
concept ExposesDescriptors = requires(T t) { t.dstMemory_; } || requires(T t) { t.bound_; };
static_assert(!ExposesBackend<mscclpp::PortChannelDeviceHandle>);
static_assert(!ExposesDescriptors<mscclpp::PortChannelDeviceHandle>);

// Same normal tutorial kernel signature/body for either host construction path.
__global__ void unifiedPortTutorial(mscclpp::PortChannelDeviceHandle channel, size_t bytes) {
  if (threadIdx.x != 0) return;
  channel.signal();
  channel.wait();
  channel.putWithSignal(0, 0, bytes);
  channel.wait();
  channel.flush();
}

void deviceTutorialFlow(mscclpp::Communicator& comm, const mscclpp::Connection& connection, int peer,
                        mscclpp::RegisteredMemory remote, mscclpp::RegisteredMemory local, size_t bytes) {
  auto sema = comm.buildSemaphore(connection, peer).get();
  mscclpp::PortChannel channel(sema, remote, local);
  unifiedPortTutorial<<<1, 1>>>(channel.deviceHandle(), bytes);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
}

void hostTutorialFlow(mscclpp::Communicator& comm, const mscclpp::Connection& connection, int peer,
                      mscclpp::RegisteredMemory remote, mscclpp::RegisteredMemory local, size_t bytes) {
  auto sema = comm.buildSemaphore(connection, peer).get();
  mscclpp::ProxyService proxy;
  auto sid = proxy.addSemaphore(sema);
  auto src = proxy.addMemory(local);
  auto dst = proxy.addMemory(remote);
  auto channel = proxy.portChannel(sid, dst, src);
  proxy.startProxy();
  unifiedPortTutorial<<<1, 1>>>(channel.deviceHandle(), bytes);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  proxy.stopProxy();
}

using ProbeConcurrency = mscclpp::PortChannelDeviceHandle::Concurrency;
template <ProbeConcurrency Mode, uint32_t Slots>
__global__ void operationScratch(mscclpp::PortChannelDeviceHandle channel) {
  channel.signal<Mode, Slots>();
  channel.flush<Mode, Slots>();
}
template __global__ void operationScratch<ProbeConcurrency::Exclusive, 0>(mscclpp::PortChannelDeviceHandle);
template __global__ void operationScratch<ProbeConcurrency::Exclusive, 8>(mscclpp::PortChannelDeviceHandle);
template __global__ void operationScratch<ProbeConcurrency::Exclusive, 32>(mscclpp::PortChannelDeviceHandle);
// Check the API upper bound without demanding a 64-KiB static scratch array from every target GPU.
template __global__ void operationScratch<ProbeConcurrency::Device, 1024>(mscclpp::PortChannelDeviceHandle);
#ifdef PROBE_INVALID_SLOTS
template __global__ void operationScratch<ProbeConcurrency::Device, 1025>(mscclpp::PortChannelDeviceHandle);
#endif

// Optional isolated negative executions, never run within the unit-test CUDA context.
// 0: default; 1: Host/Exclusive; 2: unavailable Device; 3/4: both cross-kind base assignments.
#ifdef PROBE_PORT_FAILURE
#include "port_channel_handle_access.hpp"
static_assert(PROBE_PORT_FAILURE >= 0 && PROBE_PORT_FAILURE <= 4);
#if PROBE_PORT_FAILURE >= 3
#ifndef PROBE_PORT_OPERATION
#define PROBE_PORT_OPERATION 0
#endif
__global__ void expectedPortFailure(mscclpp::PortChannelDeviceHandle channel) {
#if PROBE_PORT_FAILURE == 3
  static_assert(MSCCLPP_DETAIL_GPUNETIO_DEVICE_AVAILABLE,
                "Use a supported IB-on build/architecture so backend-unavailable rejection cannot mask this probe");
#endif
#if PROBE_PORT_OPERATION == 0
  channel.put(0, 0, 8);
#elif PROBE_PORT_OPERATION == 1
  channel.putWithSignal(0, 0, 8);
#elif PROBE_PORT_OPERATION == 2
  channel.putWithSignalAndFlush(0, 0, 8);
#elif PROBE_PORT_OPERATION == 3
  channel.accumulate(0, 1);
#elif PROBE_PORT_OPERATION == 4
  channel.putWithAccumulate(0, 0, 8, 0, 1);
#else
#error "Unknown bound-operation failure probe"
#endif
}
#else
__global__ void expectedPortFailure(mscclpp::BasePortChannelDeviceHandle channel) {
#if PROBE_PORT_FAILURE == 1
  channel.signal<mscclpp::BasePortChannelDeviceHandle::Concurrency::Exclusive>();
#else
  channel.signal();
#endif
}
#endif
int main() {
  using Base = mscclpp::BasePortChannelDeviceHandle;
  using Bound = mscclpp::PortChannelDeviceHandle;
  using Access = mscclpp::detail::PortChannelHandleAccess;
#if PROBE_PORT_FAILURE >= 3
  Bound handle;
#else
  Base handle;
#endif
#if PROBE_PORT_FAILURE == 1
  handle = Base(0, {}, {}, nullptr);
#elif PROBE_PORT_FAILURE == 2
  static_assert(!MSCCLPP_HAS_GPUNETIO, "Run the unavailable-Device probe against the IB-off installation");
  handle = Access::unavailableDeviceForTest();
#elif PROBE_PORT_FAILURE == 3
  handle = Bound(0, {}, {}, 11, 13, nullptr);
  static_cast<Base&>(handle) = Access::unavailableDeviceForTest();
#elif PROBE_PORT_FAILURE == 4
  handle = Access::unavailableBoundDeviceForTest();
  static_cast<Base&>(handle) = Base(0, {}, {}, nullptr);
#endif
  expectedPortFailure<<<1, 1>>>(handle);
  if (cudaGetLastError() != cudaSuccess) return 2;
  auto result = cudaDeviceSynchronize();
  if (result == cudaSuccess) return 1;
#if PROBE_PORT_FAILURE >= 3
  std::fprintf(stderr, "Bound-mode probe observed: %s\n", cudaGetErrorString(result));
#if defined(MSCCLPP_DEVICE_CUDA)
  // A null FIFO/QP access is not the expected tag guard. Accept trap-class errors only.
  if (result != cudaErrorIllegalInstruction && result != cudaErrorLaunchFailure) return 3;
#endif
#else
  std::fprintf(stderr, "Expected PortChannel failure: %s\n", cudaGetErrorString(result));
#endif
  return 0;
}
#endif
