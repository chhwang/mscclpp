// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_PORT_CHANNEL_HANDLE_ACCESS_HPP_
#define MSCCLPP_PORT_CHANNEL_HANDLE_ACCESS_HPP_

#include <mscclpp/errors.hpp>
#include <mscclpp/port_channel_device.hpp>

namespace mscclpp::detail {

// Host construction/binding/test access only. Not installed or part of the device API.
struct PortChannelHandleAccess {
  static BasePortChannelDeviceHandle unavailableDeviceForTest() {
    return BasePortChannelDeviceHandle(Host2DeviceSemaphoreDeviceHandle{}, GpuNetIoDeviceHandle{},
                                       GpuNetIoMemoryDeviceHandle{});
  }
  static PortChannelDeviceHandle unavailableBoundDeviceForTest() {
    return PortChannelDeviceHandle(unavailableDeviceForTest(),
                                   PortChannelDeviceHandle::DeviceBound{GpuNetIoMemoryDeviceHandle{0x1000, 17},
                                                                        GpuNetIoMemoryDeviceHandle{0x2000, 19}});
  }
  static bool hasMatchingBounds(const PortChannelDeviceHandle& handle) { return handle.hasMatchingBounds(); }
  static void requireBoundModeForTest(const PortChannelDeviceHandle& handle) { handle.requireBoundMode(); }
  static bool isDevice(const BasePortChannelDeviceHandle& handle) {
    return handle.kind_ == BasePortChannelDeviceHandle::Kind::Device;
  }
  static auto& host(BasePortChannelDeviceHandle& handle) {
    if (isDevice(handle)) throw Error("Host fields are not available on a Device handle", ErrorCode::InvalidUsage);
    handle.kind_ = BasePortChannelDeviceHandle::Kind::Host;
    return handle.state_.host;
  }
  static const auto& host(const BasePortChannelDeviceHandle& handle) {
    if (isDevice(handle)) throw Error("Host fields are not available on a Device handle", ErrorCode::InvalidUsage);
    return handle.state_.host;
  }
  static auto& hostReceive(BasePortChannelDeviceHandle& handle) {
    host(handle);
    return handle.receive_;
  }
  static const auto& receive(const BasePortChannelDeviceHandle& handle) { return handle.receive_; }
  static const auto& device(const BasePortChannelDeviceHandle& handle) {
    if (!isDevice(handle)) throw Error("Expected a Device handle", ErrorCode::InvalidUsage);
    return handle.state_.device;
  }
  static auto& hostBound(PortChannelDeviceHandle& handle) {
    using Kind = BasePortChannelDeviceHandle::Kind;
    // Only the default-derived state may be promoted by Python's Host field setters.
    if (handle.boundKind_ == Kind::Invalid && (handle.kind_ == Kind::Invalid || handle.kind_ == Kind::Host)) {
      host(handle);
      handle.boundKind_ = Kind::Host;
    }
    if (!hasMatchingBounds(handle) || handle.kind_ != Kind::Host) {
      throw Error("Host bound fields require matching Host/base state", ErrorCode::InvalidUsage);
    }
    return handle.bound_.host;
  }
  static const auto& hostBound(const PortChannelDeviceHandle& handle) {
    if (!hasMatchingBounds(handle) || isDevice(handle)) {
      throw Error("Host bound fields require matching Host/base state", ErrorCode::InvalidUsage);
    }
    return handle.bound_.host;
  }
  static const auto& deviceBound(const PortChannelDeviceHandle& handle) {
    if (!hasMatchingBounds(handle) || !isDevice(handle)) {
      throw Error("Device bound fields require matching Device/base state", ErrorCode::InvalidUsage);
    }
    return handle.bound_.device;
  }
};

}  // namespace mscclpp::detail

#endif  // MSCCLPP_PORT_CHANNEL_HANDLE_ACCESS_HPP_
