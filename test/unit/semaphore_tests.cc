// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/memory_channel.hpp>
#include <mscclpp/port_channel.hpp>
#include <type_traits>
#include <utility>

#include "../framework.hpp"
#include "connection.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
static_assert(std::is_same_v<mscclpp::MemoryDevice2DeviceSemaphore, mscclpp::Device2DeviceSemaphore>);
static_assert(std::is_same_v<mscclpp::SmDevice2DeviceSemaphore, mscclpp::Device2DeviceSemaphore>);
static_assert(
    std::is_constructible_v<mscclpp::BaseMemoryChannel, std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>>);
static_assert(std::is_constructible_v<mscclpp::MemoryChannel, std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>,
                                      mscclpp::RegisteredMemory, mscclpp::RegisteredMemory>);
static_assert(
    std::is_constructible_v<mscclpp::BasePortChannel, std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>>);
#pragma GCC diagnostic pop

static_assert(!std::is_copy_constructible_v<mscclpp::Device2DeviceSemaphore>);
static_assert(!std::is_copy_assignable_v<mscclpp::Device2DeviceSemaphore>);
static_assert(std::is_nothrow_move_constructible_v<mscclpp::Device2DeviceSemaphore>);
static_assert(std::is_nothrow_move_assignable_v<mscclpp::Device2DeviceSemaphore>);
static_assert(std::is_same_v<mscclpp::DeviceHandle<mscclpp::Device2DeviceSemaphore>,
                             mscclpp::MemoryDevice2DeviceSemaphoreDeviceHandle>);
static_assert(std::is_same_v<decltype(std::declval<const mscclpp::Device2DeviceSemaphore&>().portDeviceHandle()),
                             mscclpp::PortDevice2DeviceSemaphoreDeviceHandle>);

namespace {

// Metadata-only connection: exercises extraction/validation with real GPU counters, without a NIC or network jobs.
class SemaphoreMetadataConnection : public mscclpp::BaseConnection {
 public:
  SemaphoreMetadataConnection(std::shared_ptr<mscclpp::Context> context, const mscclpp::Endpoint& endpoint)
      : BaseConnection(std::move(context), endpoint), deviceMode(false), remoteToken{0x1000, 17} {}

  bool deviceMode;
  mscclpp::DeviceMemoryHandle remoteToken;

  bool isDeviceBackend() const override { return deviceMode; }
  mscclpp::DeviceMemoryHandle getDeviceMemoryInfo(const mscclpp::RegisteredMemory&) const override {
    return remoteToken;
  }
  mscclpp::Transport transport() const override { return mscclpp::Transport::Ethernet; }
  mscclpp::Transport remoteTransport() const override { return transport(); }
  void write(mscclpp::RegisteredMemory, uint64_t, mscclpp::RegisteredMemory, uint64_t, uint64_t) override {
    throw mscclpp::Error("metadata-only connection", mscclpp::ErrorCode::InvalidUsage);
  }
  void updateAndSync(mscclpp::RegisteredMemory, uint64_t, uint64_t*, uint64_t) override {
    throw mscclpp::Error("metadata-only connection", mscclpp::ErrorCode::InvalidUsage);
  }
  void accumulate(mscclpp::RegisteredMemory, uint64_t, int64_t) override {
    throw mscclpp::Error("metadata-only connection", mscclpp::ErrorCode::InvalidUsage);
  }
  void flush(int64_t) override { throw mscclpp::Error("metadata-only connection", mscclpp::ErrorCode::InvalidUsage); }
};

template <typename Func>
bool rejectsInvalidUsage(Func func, const char* message) {
  try {
    func();
  } catch (const mscclpp::Error& e) {
    return e.getErrorCode() == mscclpp::ErrorCode::InvalidUsage &&
           std::string(e.what()).find(message) != std::string::npos;
  }
  return false;
}

}  // namespace

class DeviceSemaphoreTest : public mscclpp::test::TestCase {
 protected:
  void SetUp() override {
    MSCCLPP_CUDATHROW(cudaSetDevice(0));
    context = mscclpp::Context::create();
    semaphore = makeSemaphore({mscclpp::DeviceType::GPU, 0});
  }

  mscclpp::Semaphore makeSemaphore(mscclpp::Device device) {
    mscclpp::EndpointConfig config;
    // Unknown avoids creating a real transport endpoint; the test connection only registers memory metadata.
    config.transport = mscclpp::Transport::Unknown;
    config.device = device;
    auto endpoint = context->createEndpoint(config);
    connectionImpl = std::make_shared<SemaphoreMetadataConnection>(context, endpoint);
    mscclpp::Connection connection;
    mscclpp::BaseConnection::getImpl(connection) = connectionImpl;
    mscclpp::SemaphoreStub stub(connection);
    return mscclpp::Semaphore(stub, stub);
  }

  std::shared_ptr<mscclpp::Context> context;
  std::shared_ptr<SemaphoreMetadataConnection> connectionImpl;
  mscclpp::Semaphore semaphore;
};

TEST(DeviceSemaphoreTest, ExtractionModesAndRemoteValidation) {
  auto owner = std::make_shared<mscclpp::Device2DeviceSemaphore>(semaphore);
  const auto memoryHandle = owner->deviceHandle();
  EXPECT_EQ(mscclpp::deviceHandle(*owner).expectedInboundToken, memoryHandle.expectedInboundToken);
  EXPECT_TRUE(rejectsInvalidUsage([&] { owner->portDeviceHandle(); }, "device-mode connection"));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::BasePortChannel invalid(owner); }, "device-mode connection"));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::PortChannel invalid(owner, {}, {}); }, "device-mode connection"));

  connectionImpl->deviceMode = true;
  EXPECT_TRUE(rejectsInvalidUsage([&] { owner->deviceHandle(); }, "use portDeviceHandle()"));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle(*owner); }, "use portDeviceHandle()"));
  for (uint64_t invalidAddress : {uint64_t{0}, uint64_t{0x1001}}) {
    connectionImpl->remoteToken.addr = invalidAddress;
    EXPECT_TRUE(rejectsInvalidUsage([&] { owner->portDeviceHandle(); }, "not addressable"));
    EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::BasePortChannel invalid(owner); }, "not addressable"));
    EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::PortChannel invalid(owner, {}, {}); }, "not addressable"));
  }
  connectionImpl->remoteToken = {0x2000, 23};
  mscclpp::BasePortChannel channel(owner);
  auto first = channel.deviceBackendHandle();
  EXPECT_EQ(first.semaphore_.expectedInboundToken, memoryHandle.expectedInboundToken);
  EXPECT_EQ(first.semaphore_.remoteInboundToken.addr, uint64_t{0x2000});
  EXPECT_EQ(first.semaphore_.remoteInboundToken.key, 23u);
  connectionImpl->remoteToken = {0x3000, 29};
  auto second = channel.deviceBackendHandle();
  EXPECT_EQ(second.semaphore_.remoteInboundToken.addr, uint64_t{0x3000});
  EXPECT_EQ(second.semaphore_.remoteInboundToken.key, 29u);
  connectionImpl->remoteToken.addr = 0;
  EXPECT_TRUE(rejectsInvalidUsage([&] { channel.deviceBackendHandle(); }, "not addressable"));
  connectionImpl->deviceMode = false;
  EXPECT_TRUE(rejectsInvalidUsage([&] { channel.deviceBackendHandle(); }, "device-mode connection"));
}

TEST(DeviceSemaphoreTest, SharedOwnershipAndCounter) {
  auto owner = std::make_shared<mscclpp::Device2DeviceSemaphore>(semaphore);
  std::weak_ptr<mscclpp::Device2DeviceSemaphore> weakOwner = owner;
  auto expected = owner->deviceHandle().expectedInboundToken;
  const uint64_t generation = 7;
  mscclpp::gpuMemcpy(expected, &generation, 1, cudaMemcpyHostToDevice);
  mscclpp::BaseMemoryChannel memoryChannel(owner);
  auto memoryCopy = memoryChannel;
  EXPECT_EQ(memoryCopy.deviceHandle().semaphore_.expectedInboundToken, expected);

  connectionImpl->deviceMode = true;
  mscclpp::PortChannel channel(owner, semaphore.remoteMemory(), semaphore.localMemory());
  auto portCopy = channel;
  mscclpp::BasePortChannel baseCopy = channel;
  owner.reset();
  channel = {};
  mscclpp::BaseMemoryChannel emptyMemory;
  memoryChannel = emptyMemory;
  memoryCopy = emptyMemory;
  EXPECT_FALSE(weakOwner.expired());
  EXPECT_EQ(portCopy.deviceBackendHandle().semaphore_.expectedInboundToken, expected);
  EXPECT_EQ(baseCopy.deviceBackendHandle().semaphore_.expectedInboundToken, expected);
  uint64_t actual = 0;
  mscclpp::gpuMemcpy(&actual, expected, 1, cudaMemcpyDeviceToHost);
  EXPECT_EQ(actual, generation);
  portCopy = {};
  EXPECT_FALSE(weakOwner.expired());
  baseCopy = {};
  EXPECT_TRUE(weakOwner.expired());
}

TEST(DeviceSemaphoreTest, MoveRetainsCounter) {
  mscclpp::Device2DeviceSemaphore original(semaphore);
  auto expected = original.deviceHandle().expectedInboundToken;
  mscclpp::Device2DeviceSemaphore moved(std::move(original));
  EXPECT_EQ(moved.deviceHandle().expectedInboundToken, expected);
  mscclpp::Device2DeviceSemaphore assigned(semaphore);
  assigned = std::move(moved);
  EXPECT_EQ(assigned.deviceHandle().expectedInboundToken, expected);
  connectionImpl->deviceMode = true;
  EXPECT_EQ(assigned.portDeviceHandle().expectedInboundToken, expected);
}

TEST(DeviceSemaphoreTest, CpuEndpointRejected) {
  auto cpuSemaphore = makeSemaphore({mscclpp::DeviceType::CPU, 0});
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::Device2DeviceSemaphore invalid(cpuSemaphore); }, "should be GPU"));
}

TEST(DeviceSemaphoreTest, AllocatesOnEndpointGpuAndRestoresCaller) {
  int count = 0;
  MSCCLPP_CUDATHROW(cudaGetDeviceCount(&count));
  if (count < 2) SKIP_TEST() << "Requires two visible GPUs to check caller-device restoration";
  mscclpp::CudaDeviceGuard callerDevice(1);
  for (bool deviceMode : {false, true}) {
    connectionImpl->deviceMode = deviceMode;
    mscclpp::Device2DeviceSemaphore owner(semaphore);
    int current = -1;
    MSCCLPP_CUDATHROW(cudaGetDevice(&current));
    EXPECT_EQ(current, 1);
    auto expected =
        deviceMode ? owner.portDeviceHandle().expectedInboundToken : owner.deviceHandle().expectedInboundToken;
    int allocatedDevice = -1;
    MSCCLPP_CUTHROW(cuPointerGetAttribute(&allocatedDevice, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                          reinterpret_cast<CUdeviceptr>(expected)));
    EXPECT_EQ(allocatedDevice, 0);
  }
}
