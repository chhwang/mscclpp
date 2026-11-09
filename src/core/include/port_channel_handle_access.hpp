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
    host(handle);
    return handle.bound_.host;
  }
  static const auto& hostBound(const PortChannelDeviceHandle& handle) {
    host(handle);
    return handle.bound_.host;
  }
  static const auto& deviceBound(const PortChannelDeviceHandle& handle) {
    device(handle);
    return handle.bound_.device;
  }
};

}  // namespace mscclpp::detail

#endif  // MSCCLPP_PORT_CHANNEL_HANDLE_ACCESS_HPP_
