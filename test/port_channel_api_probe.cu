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
// 0: default handle; 1: Host/Exclusive; 2: Device on an IB-off/unsupported build.
#ifdef PROBE_PORT_FAILURE
#include "port_channel_handle_access.hpp"
__global__ void expectedPortFailure(mscclpp::BasePortChannelDeviceHandle channel) {
#if PROBE_PORT_FAILURE == 1
  channel.signal<mscclpp::BasePortChannelDeviceHandle::Concurrency::Exclusive>();
#else
  channel.signal();
#endif
}
int main() {
  mscclpp::BasePortChannelDeviceHandle handle;
#if PROBE_PORT_FAILURE == 1
  handle = mscclpp::BasePortChannelDeviceHandle(0, {}, {}, nullptr);
#elif PROBE_PORT_FAILURE == 2
  static_assert(!MSCCLPP_HAS_GPUNETIO, "Run the unavailable-Device probe against the IB-off installation");
  handle = mscclpp::detail::PortChannelHandleAccess::unavailableDeviceForTest();
#endif
  expectedPortFailure<<<1, 1>>>(handle);
  if (cudaGetLastError() != cudaSuccess) return 2;
  auto result = cudaDeviceSynchronize();
  if (result == cudaSuccess) return 1;
  std::fprintf(stderr, "Expected PortChannel failure: %s\n", cudaGetErrorString(result));
  return 0;
}
#endif
