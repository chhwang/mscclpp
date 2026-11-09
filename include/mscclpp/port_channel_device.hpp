// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_PORT_CHANNEL_DEVICE_HPP_
#define MSCCLPP_PORT_CHANNEL_DEVICE_HPP_

#include <type_traits>

#include "detail/gpunetio_device.hpp"
#include "fifo_device.hpp"
#include "semaphore_device.hpp"

namespace mscclpp {

/// Where a PortChannel's operations are serviced.
enum class PortChannelBackend : uint8_t { Host = 0, Device = 1 };
/// Semaphore ID in a ProxyService registry.
using SemaphoreId = uint32_t;
/// Registered memory ID in a ProxyService registry.
using MemoryId = uint32_t;

struct BasePortChannel;
struct PortChannel;
struct PortChannelDeviceHandle;
namespace detail {
struct PortChannelHandleAccess;
}

/// Device-side PortChannel without bound source/destination regions.
/// The same trivially copyable type services Host and Device channels.
struct BasePortChannelDeviceHandle {
 public:
  /// Scope of concurrent callers on the underlying queue pair.
  enum class Concurrency : uint8_t { Exclusive = 0, Block = 1, Device = 2 };
  /// Default internal scratch capacity per CTA; zero disables scratch submission.
  static constexpr uint32_t kDefaultSmemSlots = detail::kPortChannelDefaultSmemSlots;

  /// Constructor. Operations on an unassigned handle fail explicitly.
  MSCCLPP_HOST_DEVICE_INLINE constexpr BasePortChannelDeviceHandle() : kind_(Kind::Invalid), receive_{}, state_() {}

  /// Construct a Host-proxy handle. Prefer extraction from a host channel.
  MSCCLPP_HOST_DEVICE_INLINE constexpr BasePortChannelDeviceHandle(SemaphoreId id,
                                                                   Host2DeviceSemaphoreDeviceHandle receive,
                                                                   FifoDeviceHandle fifo, uint64_t* flushDone)
      : kind_(Kind::Host), receive_(receive), state_(HostState{id, fifo, flushDone}) {}

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Put between ProxyService memory IDs. Only Host-backed handles accept IDs.
  /// Host posting supports Concurrency::Device only; incompatible modes trap even in release builds.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void put(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset, uint64_t size) {
    requireHost<Mode, SmemSlots>();
    state_.host.fifo.push({TriggerPut, dst, dstOffset, src, srcOffset, size, state_.host.id});
  }
  /// Put between Host memory IDs at the same offset.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void put(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    put<Mode, SmemSlots>(dst, offset, src, offset, size);
  }
  /// Signal the remote peer. Scratch capacity is selected per operation, not by handle type.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void signal(int slotIdx = -1) {
    validate<Mode, SmemSlots>();
    if (kind_ == Kind::Host) {
      state_.host.fifo.push({TriggerSignal, 0, 0, 0, 0, 0, state_.host.id});
    } else {
      state_.device.backend.template accumulate<Mode, SmemSlots>(state_.device.remoteSignal, 0, 1, slotIdx);
    }
  }
  /// Put between Host memory IDs and signal in FIFO order.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignal(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                                           uint64_t size) {
    requireHost<Mode, SmemSlots>();
    state_.host.fifo.push({TriggerPutWithSignal, dst, dstOffset, src, srcOffset, size, state_.host.id});
  }
  /// Put and signal between Host memory IDs at the same offset.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignal(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    putWithSignal<Mode, SmemSlots>(dst, offset, src, offset, size);
  }
  /// Put, signal and flush between Host memory IDs with a single FIFO trigger.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                                                   uint64_t size, int64_t maxSpinCount = 1000000) {
    requireHost<Mode, SmemSlots>();
    auto pos =
        state_.host.fifo.push({TriggerPutWithSignalAndFlush, dst, dstOffset, src, srcOffset, size, state_.host.id});
    waitFlush(pos, maxSpinCount);
  }
  /// Put, signal and flush between Host memory IDs at the same offset.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size,
                                                   int64_t maxSpinCount = 1000000) {
    putWithSignalAndFlush<Mode, SmemSlots>(dst, offset, src, offset, size, maxSpinCount);
  }
  /// Wait for local completion of operations posted so far on this connection.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void flush(int64_t maxSpinCount = 1000000) {
    validate<Mode, SmemSlots>();
    if (kind_ == Kind::Host) {
      auto pos = state_.host.fifo.push({TriggerFlush, 0, 0, 0, 0, 0, state_.host.id});
      waitFlush(pos, maxSpinCount);
    } else {
      state_.device.backend.template flush<Mode>(maxSpinCount);
    }
  }
  /// Atomically add to a Host-registered destination. See Connection::accumulate for transport support.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void accumulate(MemoryId dst, uint64_t offset, int64_t value) {
    requireHost<Mode, SmemSlots>();
    uint64_t operand = static_cast<uint64_t>(value);
    state_.host.fifo.push({TriggerAccumulate, dst, offset, 0, operand >> TriggerBitsSize,
                           static_cast<uint32_t>(operand), state_.host.id});
  }
  /// Poll the shared receive counter, consuming one expected signal when available.
  MSCCLPP_DEVICE_INLINE bool poll() {
    requireReceive();
    return receive_.poll();
  }
  /// Wait for a remote signal.
  MSCCLPP_DEVICE_INLINE void wait(int64_t maxSpinCount = 10000000) {
    requireReceive();
    receive_.wait(maxSpinCount);
  }
#endif

 private:
  enum class Kind : uint8_t { Invalid, Host, Device };
  struct HostState {
    SemaphoreId id;
    FifoDeviceHandle fifo;
    uint64_t* flushDone;
  };
  struct DeviceState {
    detail::GpuNetIoDeviceHandle backend;
    detail::GpuNetIoMemoryDeviceHandle remoteSignal;
  };
  union State {
    HostState host;
    DeviceState device;
    MSCCLPP_HOST_DEVICE_INLINE constexpr State() : host{} {}
    MSCCLPP_HOST_DEVICE_INLINE constexpr explicit State(HostState value) : host(value) {}
    MSCCLPP_HOST_DEVICE_INLINE constexpr explicit State(DeviceState value) : device(value) {}
  };
  Kind kind_;
  Host2DeviceSemaphoreDeviceHandle receive_;
  State state_;

  MSCCLPP_HOST_DEVICE_INLINE constexpr BasePortChannelDeviceHandle(Host2DeviceSemaphoreDeviceHandle receive,
                                                                   detail::GpuNetIoDeviceHandle backend,
                                                                   detail::GpuNetIoMemoryDeviceHandle remoteSignal)
      : kind_(Kind::Device), receive_(receive), state_(DeviceState{backend, remoteSignal}) {}

#if defined(MSCCLPP_DEVICE_COMPILE)
  template <Concurrency Mode, uint32_t SmemSlots>
  MSCCLPP_DEVICE_INLINE void validate() const {
    static_assert(SmemSlots <= 1024, "At most 1024 shared-memory slots per CTA are supported");
    static_assert(Mode == Concurrency::Exclusive || Mode == Concurrency::Block || Mode == Concurrency::Device);
    requireValid();
    if (kind_ == Kind::Host) {
      detail::portChannelRequire(Mode == Concurrency::Device, "Host channels only support Concurrency::Device");
    }
  }
  template <Concurrency Mode, uint32_t SmemSlots>
  MSCCLPP_DEVICE_INLINE void requireHost() const {
    validate<Mode, SmemSlots>();
    detail::portChannelRequire(kind_ == Kind::Host, "memory IDs require a Host-proxy channel");
  }
  MSCCLPP_DEVICE_INLINE void requireValid() const {
    detail::portChannelRequire(kind_ == Kind::Host || kind_ == Kind::Device, "unassigned channel handle");
#if !MSCCLPP_DETAIL_GPUNETIO_DEVICE_AVAILABLE
    detail::portChannelRequire(kind_ != Kind::Device, "Device backend is not available for this build/architecture");
#endif
  }
  MSCCLPP_DEVICE_INLINE void requireReceive() const {
    requireValid();
    detail::portChannelRequire(receive_.inboundToken && receive_.expectedInboundToken,
                               "unassigned channel receive state");
  }
  MSCCLPP_DEVICE_INLINE void waitFlush(uint64_t pos, [[maybe_unused]] int64_t maxSpinCount) {
    detail::portChannelRequire(state_.host.flushDone != nullptr, "missing Host flush state");
    POLL_MAYBE_JAILBREAK((atomicLoad<uint64_t, scopeSystem>(state_.host.flushDone, memoryOrderAcquire) <= pos),
                         maxSpinCount);
  }
  template <Concurrency Mode, uint32_t SmemSlots>
  MSCCLPP_DEVICE_INLINE void putDevice(const detail::GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                       const detail::GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset, uint64_t size,
                                       int slotIdx) {
    validate<Mode, SmemSlots>();
    state_.device.backend.template put<Mode, SmemSlots>(dst, dstOffset, src, srcOffset, size, slotIdx);
  }
  template <Concurrency Mode, uint32_t SmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignalDevice(const detail::GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                                 const detail::GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset,
                                                 uint64_t size, int slotIdx) {
    validate<Mode, SmemSlots>();
    state_.device.backend.template putWithAccumulate<Mode, SmemSlots>(dst, dstOffset, src, srcOffset, size,
                                                                      state_.device.remoteSignal, 0, 1, slotIdx);
  }
#endif
  friend struct BasePortChannel;
  friend struct PortChannel;
  friend struct PortChannelDeviceHandle;
  friend struct detail::PortChannelHandleAccess;
};

/// Bound PortChannel device handle. The same normal kernel parameter type supports both backends.
struct PortChannelDeviceHandle : public BasePortChannelDeviceHandle {
 public:
  /// Constructor. Operations on an unassigned handle fail explicitly.
  MSCCLPP_HOST_DEVICE_INLINE constexpr PortChannelDeviceHandle() : BasePortChannelDeviceHandle(), bound_() {}
  /// Construct a bound Host-proxy handle. Prefer host-channel extraction.
  MSCCLPP_HOST_DEVICE_INLINE constexpr PortChannelDeviceHandle(SemaphoreId id, Host2DeviceSemaphoreDeviceHandle receive,
                                                               FifoDeviceHandle fifo, MemoryId dst, MemoryId src,
                                                               uint64_t* flushDone)
      : BasePortChannelDeviceHandle(id, receive, fifo, flushDone), bound_(HostBound{dst, src}) {}

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Put between the bound regions. slotIdx=-1 uses the linear CTA thread index for internal scratch.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t size, int slotIdx = -1) {
    validate<Mode, SmemSlots>();
    if (kind_ == Kind::Host) {
      BasePortChannelDeviceHandle::put<Mode, SmemSlots>(bound_.host.dst, dstOffset, bound_.host.src, srcOffset, size);
    } else {
      putDevice<Mode, SmemSlots>(bound_.device.dst, dstOffset, bound_.device.src, srcOffset, size, slotIdx);
    }
  }
  /// Put at the same offset in both bound regions.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void put(uint64_t offset, uint64_t size) {
    put<Mode, SmemSlots>(offset, offset, size);
  }
  /// Put and signal the remote peer.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t dstOffset, uint64_t srcOffset, uint64_t size, int slotIdx = -1) {
    validate<Mode, SmemSlots>();
    if (kind_ == Kind::Host) {
      BasePortChannelDeviceHandle::putWithSignal<Mode, SmemSlots>(bound_.host.dst, dstOffset, bound_.host.src,
                                                                  srcOffset, size);
    } else {
      putWithSignalDevice<Mode, SmemSlots>(bound_.device.dst, dstOffset, bound_.device.src, srcOffset, size, slotIdx);
    }
  }
  /// Put and signal at the same offset in both regions.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t offset, uint64_t size) {
    putWithSignal<Mode, SmemSlots>(offset, offset, size);
  }
  /// Put, signal and wait for local completion.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
                                                   int64_t maxSpinCount = 1000000, int slotIdx = -1) {
    validate<Mode, SmemSlots>();
    if (kind_ == Kind::Host) {
      BasePortChannelDeviceHandle::putWithSignalAndFlush<Mode, SmemSlots>(bound_.host.dst, dstOffset, bound_.host.src,
                                                                          srcOffset, size, maxSpinCount);
    } else {
      putWithSignalDevice<Mode, SmemSlots>(bound_.device.dst, dstOffset, bound_.device.src, srcOffset, size, slotIdx);
      flush<Mode, SmemSlots>(maxSpinCount);
    }
  }
  /// Put, signal and flush at a common offset.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(uint64_t offset, uint64_t size) {
    putWithSignalAndFlush<Mode, SmemSlots>(offset, offset, size);
  }
  /// Atomically add to the bound destination region.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void accumulate(uint64_t offset, int64_t value, int slotIdx = -1) {
    validate<Mode, SmemSlots>();
    if (kind_ == Kind::Host) {
      BasePortChannelDeviceHandle::accumulate<Mode, SmemSlots>(bound_.host.dst, offset, value);
    } else {
      state_.device.backend.template accumulate<Mode, SmemSlots>(bound_.device.dst, offset, value, slotIdx);
    }
  }
  /// Device backend only: post a put followed by an atomic add. Observing the add publishes the put.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithAccumulate(uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
                                               uint64_t accOffset, int64_t value, int slotIdx = -1) {
    validate<Mode, SmemSlots>();
    detail::portChannelRequire(kind_ == Kind::Device, "putWithAccumulate requires a Device channel");
    state_.device.backend.template putWithAccumulate<Mode, SmemSlots>(
        bound_.device.dst, dstOffset, bound_.device.src, srcOffset, size, bound_.device.dst, accOffset, value, slotIdx);
  }
#endif

 private:
  struct HostBound {
    MemoryId dst;
    MemoryId src;
  };
  struct DeviceBound {
    detail::GpuNetIoMemoryDeviceHandle dst;
    detail::GpuNetIoMemoryDeviceHandle src;
  };
  union Bound {
    HostBound host;
    DeviceBound device;
    MSCCLPP_HOST_DEVICE_INLINE constexpr Bound() : host{} {}
    MSCCLPP_HOST_DEVICE_INLINE constexpr explicit Bound(HostBound value) : host(value) {}
    MSCCLPP_HOST_DEVICE_INLINE constexpr explicit Bound(DeviceBound value) : device(value) {}
  };
  Bound bound_;
  MSCCLPP_HOST_DEVICE_INLINE constexpr PortChannelDeviceHandle(BasePortChannelDeviceHandle base, HostBound bound)
      : BasePortChannelDeviceHandle(base), bound_(bound) {}
  MSCCLPP_HOST_DEVICE_INLINE constexpr PortChannelDeviceHandle(BasePortChannelDeviceHandle base, DeviceBound bound)
      : BasePortChannelDeviceHandle(base), bound_(bound) {}
  friend struct PortChannel;
  friend struct detail::PortChannelHandleAccess;
};

static_assert(std::is_trivially_copyable_v<BasePortChannelDeviceHandle>);
static_assert(std::is_trivially_copyable_v<PortChannelDeviceHandle>);

}  // namespace mscclpp

#endif  // MSCCLPP_PORT_CHANNEL_DEVICE_HPP_
