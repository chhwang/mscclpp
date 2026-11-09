// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SEMAPHORE_HPP_
#define MSCCLPP_SEMAPHORE_HPP_

#include <memory>

#include "core.hpp"
#include "gpu_utils.hpp"
#include "semaphore_device.hpp"

namespace mscclpp {

/// A semaphore for sending signals from the host to the device.
class Host2DeviceSemaphore {
 private:
  Semaphore semaphore_;
  std::shared_ptr<uint64_t> inboundToken_;
  detail::UniqueGpuPtr<uint64_t> expectedInboundToken_;
  std::unique_ptr<uint64_t> outboundToken_;

 public:
  /// Constructor.
  /// @param semaphore
  Host2DeviceSemaphore(const Semaphore& semaphore);

  /// Constructor.
  /// @param communicator The communicator.
  /// @param connection The connection associated with this semaphore.
  Host2DeviceSemaphore(Communicator& communicator, const Connection& connection);

  /// Destructor.
  ~Host2DeviceSemaphore();

  /// Move constructor.
  Host2DeviceSemaphore(Host2DeviceSemaphore&&) noexcept = default;

  /// Move assignment operator.
  Host2DeviceSemaphore& operator=(Host2DeviceSemaphore&&) noexcept = default;

  /// Returns the connection.
  /// @return The connection associated with this semaphore.
  Connection& connection();

  /// Signal the device.
  void signal();

  /// Device-side handle for Host2DeviceSemaphore.
  using DeviceHandle = Host2DeviceSemaphoreDeviceHandle;

  /// Returns the device-side handle.
  DeviceHandle deviceHandle() const;
};

/// A semaphore for sending signals from the local host to a remote host.
class Host2HostSemaphore {
 private:
  Semaphore semaphore_;
  std::unique_ptr<uint64_t> expectedInboundToken_;
  std::unique_ptr<uint64_t> outboundToken_;

 public:
  /// Constructor.
  /// @param semaphore
  Host2HostSemaphore(const Semaphore& semaphore);

  /// Constructor.
  /// @param communicator The communicator.
  /// @param connection The connection associated with this semaphore. Transport::CudaIpc is not allowed for
  /// Host2HostSemaphore.
  Host2HostSemaphore(Communicator& communicator, const Connection& connection);

  /// Returns the connection.
  /// @return The connection associated with this semaphore.
  Connection& connection();

  /// Signal the remote host.
  void signal();

  /// Check if the remote host has signaled.
  /// @return true if the remote host has signaled.
  bool poll();

  /// Wait for the remote host to signal.
  /// @param maxSpinCount The maximum number of spin counts before throwing an exception. Never throws if negative.
  void wait(int64_t maxSpinCount = 10000000);
};

/// A move-only semaphore for signaling a peer device through memory or a device-backend PortChannel.
class Device2DeviceSemaphore {
 private:
  Semaphore semaphore_;
  detail::UniqueGpuPtr<uint64_t> expectedInboundToken_;

  Host2DeviceSemaphoreDeviceHandle portReceiveHandle() const;
  friend struct BasePortChannel;

 public:
  /// Constructor. Allocates the expected counter on the local endpoint's GPU and restores the caller's device.
  /// @param semaphore The semaphore associated with the connection.
  /// @throws Error with ErrorCode::InvalidUsage if the local endpoint is not a GPU.
  Device2DeviceSemaphore(const Semaphore& semaphore);

  /// Constructor.
  /// @param communicator The communicator.
  /// @param connection The connection associated with this semaphore.
  Device2DeviceSemaphore(Communicator& communicator, const Connection& connection);

  /// Returns the connection.
  /// @return The connection associated with this semaphore.
  Connection& connection();

  /// Memory-facing device-side handle for Device2DeviceSemaphore.
  using DeviceHandle = MemoryDevice2DeviceSemaphoreDeviceHandle;

  /// Returns the memory-facing device-side handle.
  /// @throws Error with ErrorCode::InvalidUsage for a PortChannel device-mode connection.
  DeviceHandle deviceHandle() const;
};

/// @deprecated Use Device2DeviceSemaphore instead.
using MemoryDevice2DeviceSemaphore [[deprecated("Use Device2DeviceSemaphore instead.")]] = Device2DeviceSemaphore;

/// @deprecated Use Device2DeviceSemaphore instead.
using SmDevice2DeviceSemaphore [[deprecated("Use Device2DeviceSemaphore instead.")]] = Device2DeviceSemaphore;

}  // namespace mscclpp

#endif  // MSCCLPP_SEMAPHORE_HPP_
