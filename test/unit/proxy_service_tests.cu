// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <chrono>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/port_channel.hpp>
#include <thread>
#include <type_traits>
#include <vector>

#include "../framework.hpp"

static_assert(std::is_nothrow_default_constructible_v<mscclpp::BasePortChannel>);
static_assert(std::is_nothrow_default_constructible_v<mscclpp::PortChannel>);
static_assert(std::is_same_v<mscclpp::DeviceHandle<mscclpp::BasePortChannel>, mscclpp::BasePortChannelDeviceHandle>);
static_assert(std::is_same_v<mscclpp::DeviceHandle<mscclpp::PortChannel>, mscclpp::PortChannelDeviceHandle>);
static_assert(std::is_same_v<decltype(&mscclpp::BasePortChannel::deviceHandle),
                             mscclpp::BasePortChannelDeviceHandle (mscclpp::BasePortChannel::*)() const>);
static_assert(std::is_same_v<decltype(&mscclpp::PortChannel::deviceHandle),
                             mscclpp::PortChannelDeviceHandle (mscclpp::PortChannel::*)() const>);

namespace {
template <typename Func>
bool rejectsInvalidUsage(Func func) {
  try {
    func();
  } catch (const mscclpp::Error& e) {
    return e.getErrorCode() == mscclpp::ErrorCode::InvalidUsage;
  }
  return false;
}
}  // namespace

TEST(ProxyServiceTest, InvalidChannels) {
  mscclpp::BasePortChannel base;
  mscclpp::PortChannel channel;
  EXPECT_TRUE(rejectsInvalidUsage([&] { base.deviceHandle(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { base.deviceBackendHandle(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { channel.deviceHandle(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { channel.deviceBackendHandle(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle(base); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle(channel); }));

  std::shared_ptr<mscclpp::Host2DeviceSemaphore> hostSemaphore;
  std::shared_ptr<mscclpp::Device2DeviceSemaphore> deviceSemaphore;
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::BasePortChannel invalid(0, hostSemaphore, nullptr); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::PortChannel invalid(0, hostSemaphore, nullptr, 0, 0); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::BasePortChannel invalid(deviceSemaphore); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::PortChannel invalid(deviceSemaphore, {}, {}); }));
}

TEST(ProxyServiceTest, HostChannelHandlesAndOwnership) {
  MSCCLPP_CUDATHROW(cudaSetDevice(0));
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(0, 1);
  bootstrap->initialize(mscclpp::TcpBootstrap::createUniqueId());
  mscclpp::Communicator communicator(bootstrap);
  auto connection = communicator.connect(mscclpp::Transport::CudaIpc, 0).get();
  auto service = std::make_unique<mscclpp::ProxyService>();
  auto id = service->buildAndAddSemaphore(communicator, connection);
  auto semaphore = service->semaphore(id);
  std::weak_ptr<mscclpp::Host2DeviceSemaphore> weakSemaphore = semaphore;
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::BasePortChannel invalid(id, semaphore, nullptr); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::PortChannel invalid(id, semaphore, nullptr, 0, 0); }));

  auto base = service->basePortChannel(id);
  auto channel = service->portChannel(id, 17, 23);
  const auto baseHandle = mscclpp::deviceHandle(base);
  const auto handle = mscclpp::deviceHandle(channel);
  EXPECT_EQ(handle.semaphoreId_, id);
  EXPECT_EQ(handle.dst_, 17u);
  EXPECT_EQ(handle.src_, 23u);
  EXPECT_EQ(baseHandle.semaphore_.expectedInboundToken, handle.semaphore_.expectedInboundToken);
  EXPECT_TRUE(rejectsInvalidUsage([&] { base.deviceBackendHandle(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { channel.deviceBackendHandle(); }));

  auto baseCopy = base;
  auto channelCopy = channel;
  base = {};
  channel = {};
  semaphore.reset();
  service.reset();
  EXPECT_FALSE(weakSemaphore.expired());
  EXPECT_EQ(baseCopy.deviceHandle().semaphore_.expectedInboundToken, baseHandle.semaphore_.expectedInboundToken);
  EXPECT_EQ(channelCopy.deviceHandle().semaphore_.expectedInboundToken, handle.semaphore_.expectedInboundToken);
  EXPECT_EQ(channelCopy.deviceHandle().dst_, handle.dst_);
  EXPECT_EQ(channelCopy.deviceHandle().src_, handle.src_);
  baseCopy = {};
  EXPECT_FALSE(weakSemaphore.expired());
  channelCopy = {};
  EXPECT_TRUE(weakSemaphore.expired());
}

__global__ void kernelProxyServiceTransfer(mscclpp::PortChannelDeviceHandle channel, size_t bytes) {
  channel.putWithSignalAndFlush(0, 0, bytes);
  channel.wait();
}

TEST(ProxyServiceTest, Lifecycle) {
  MSCCLPP_CUDATHROW(cudaSetDevice(0));
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(0, 1);
  bootstrap->initialize(mscclpp::TcpBootstrap::createUniqueId());
  mscclpp::Communicator communicator(bootstrap);
  auto connection = communicator.connect(mscclpp::Transport::CudaIpc, 0).get();
  constexpr size_t count = 1024;
  constexpr size_t bytes = count * sizeof(int);
  auto src = mscclpp::GpuBuffer(bytes).memory();
  auto dst = mscclpp::GpuBuffer(bytes).memory();
  auto srcMemory = communicator.registerMemory(src.get(), bytes, mscclpp::Transport::CudaIpc);
  auto dstMemory = communicator.registerMemory(dst.get(), bytes, mscclpp::Transport::CudaIpc);
  std::vector<int> expected(count, 777);
  std::vector<int> actual(count);
  MSCCLPP_CUDATHROW(cudaMemcpy(src.get(), expected.data(), bytes, cudaMemcpyHostToDevice));

  {
    mscclpp::ProxyService service;
    service.stopProxy();
    service.stopProxy();
  }

  for (int iteration = 0; iteration < 12; ++iteration) {
    auto service = std::make_unique<mscclpp::ProxyService>();
    auto srcId = service->addMemory(srcMemory);
    auto dstId = service->addMemory(dstMemory);
    auto semaphoreId = service->buildAndAddSemaphore(communicator, connection);
    auto original = service->portChannel(semaphoreId, dstId, srcId);
    auto channel = std::make_unique<mscclpp::PortChannel>(original);
    original = {};
    for (int run = 0; run < 2; ++run) {
      MSCCLPP_CUDATHROW(cudaMemset(dst.get(), 0, bytes));
      service->startProxy(true);
      kernelProxyServiceTransfer<<<1, 1>>>(channel->deviceHandle(), bytes);
      MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
      MSCCLPP_CUDATHROW(cudaMemcpy(actual.data(), dst.get(), bytes, cudaMemcpyDeviceToHost));
      EXPECT_TRUE(actual == expected);
      if (run == 0) {
        service->stopProxy();
        service->stopProxy();
      }
    }
    if (iteration % 3 == 0) {
      service->stopProxy();
      service->stopProxy();
      service.reset();
    } else if (iteration % 3 == 1) {
      channel.reset();
      service.reset();
    } else {
      // Retain the channel while destroying the service through its virtual base.
      std::unique_ptr<mscclpp::BaseProxyService> base = std::move(service);
      base.reset();
      // Give stale callbacks time to execute under host memory sanitizers.
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      channel.reset();
    }
  }
}
