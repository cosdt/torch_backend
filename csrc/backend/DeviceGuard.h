#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>
#include "csrc/backend/DeviceGuardImpl.h"
#include "csrc/core/guard/PrivateUse1Guard.h"

#include <cstddef>

namespace c10::backend {

// This code is kind of boilerplatey.  See Note [Whither the DeviceGuard
// boilerplate]

/// A variant of DeviceGuard that is specialized for device.  It accepts
/// integer indices (interpreting them as devices) and is a little
/// more efficient than DeviceGuard (it compiles to straight line
/// SetDevice/GetDevice calls); however, it can only be used
/// from code that links against device directly.
struct DeviceGuard : public Guard::PrivateUse1Guard<impl::DeviceGuardImpl> {
  using PrivateUse1Guard = Guard::PrivateUse1Guard<impl::DeviceGuardImpl>;
  using PrivateUse1Guard::PrivateUse1Guard;
  // Copy is not allowed
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

  // Move is not allowed (there is no uninitialized state)
  DeviceGuard(DeviceGuard&& other) = delete;
  DeviceGuard& operator=(DeviceGuard&& other) = delete;
};

/// A variant of OptionalDeviceGuard that is specialized for device.  See
/// DeviceGuard for when you can use this.
struct OptionalDeviceGuard
    : public Guard::OptionalPrivateUse1Guard<impl::DeviceGuardImpl> {
  using OptionalPrivateUse1Guard =
      Guard::OptionalPrivateUse1Guard<impl::DeviceGuardImpl>;
  using OptionalPrivateUse1Guard::OptionalPrivateUse1Guard;

  // Copy is not allowed
  OptionalDeviceGuard(const OptionalDeviceGuard&) = delete;
  OptionalDeviceGuard& operator=(const OptionalDeviceGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalDeviceGuard(OptionalDeviceGuard&& other) = delete;
  // See Note [Move assignment for RAII guards is tricky]
  OptionalDeviceGuard& operator=(OptionalDeviceGuard&& other) = delete;
};

/// A variant of StreamGuard that is specialized for device.  See DeviceGuard
/// for when you can use this.
struct StreamGuard {
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit StreamGuard() = delete;

  /// Set the current device to the device associated with the passed
  /// stream, and set the current device stream on that device to the passed
  /// stream. Errors if the Stream is not a device stream.
  explicit StreamGuard(c10::Stream stream) : guard_(stream) {}

  /// Copy is disallowed
  StreamGuard(const StreamGuard&) = delete;
  StreamGuard& operator=(const StreamGuard&) = delete;

  /// Move is disallowed, as StreamGuard does not have an uninitialized
  /// state, which is required for moves on types with nontrivial destructors.
  StreamGuard(StreamGuard&& other) = delete;
  StreamGuard& operator=(StreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Errors if the stream passed is not a stream.
  ///
  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  ///
  /// WARNING: reset_stream does NOT preserve previously set streams on
  /// different devices.  If you need to set streams on multiple devices,
  /// use MultiStreamGuard instead.
  void reset_stream(c10::Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the stream that was set at the time the guard was constructed.
  Stream original_stream() const {
    return Stream(Stream::UNCHECKED, guard_.original_stream());
  }

  /// Returns the most recent stream that was set using this device guard,
  /// either from construction, or via set_stream.
  Stream current_stream() const {
    return Stream(Stream::UNCHECKED, guard_.current_stream());
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  c10::Device current_device() const {
    return guard_.current_device();
  }

  /// Returns the device that was set at the most recent reset_stream(),
  /// or otherwise the device at construction time.
  c10::Device original_device() const {
    return guard_.original_device();
  }

 private:
  c10::impl::InlineStreamGuard<c10::backend::impl::DeviceGuardImpl> guard_;
};

/// A variant of OptionalStreamGuard that is specialized for device.  See
/// DeviceGuard for when you can use this.
struct OptionalStreamGuard {
  /// Create an uninitialized guard.
  explicit OptionalStreamGuard() : guard_() {}

  /// Set the current device device to the device associated with the passed
  /// stream, and set the current device stream on that device to the passed
  /// stream. Errors if the Stream is not a device stream.
  explicit OptionalStreamGuard(c10::Stream stream) : guard_(stream) {}

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream,
  /// if the passed stream is not nullopt.
  explicit OptionalStreamGuard(c10::optional<c10::Stream> stream_opt)
      : guard_(stream_opt) {}

  /// Copy is disallowed
  OptionalStreamGuard(const OptionalStreamGuard&) = delete;
  OptionalStreamGuard& operator=(const OptionalStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalStreamGuard(OptionalStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalStreamGuard& operator=(OptionalStreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Initializes the guard if it was not previously initialized.
  void reset_stream(c10::Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the stream that was set at the time the guard was most
  /// recently initialized, or nullopt if the guard is uninitialized.
  c10::optional<Stream> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return c10::make_optional(Stream(Stream::UNCHECKED, r.value()));
    } else {
      return c10::nullopt;
    }
  }

  /// Returns the most recent stream that was set using this stream guard,
  /// either from construction, or via reset_stream, if the guard is
  /// initialized, or nullopt if the guard is uninitialized.
  c10::optional<Stream> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return c10::make_optional(Stream(Stream::UNCHECKED, r.value()));
    } else {
      return c10::nullopt;
    }
  }

  /// Restore the original device and stream, resetting this guard to
  /// uninitialized state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalStreamGuard<c10::backend::impl::DeviceGuardImpl>
      guard_;
};

/// A variant of MultiStreamGuard that is specialized for device.
struct MultiStreamGuard {
  explicit MultiStreamGuard(at::ArrayRef<Stream> streams)
      : guard_(unwrapStreams(streams)) {}

  /// Copy is disallowed
  MultiStreamGuard(const MultiStreamGuard&) = delete;
  MultiStreamGuard& operator=(const MultiStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  MultiStreamGuard(MultiStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  MultiStreamGuard& operator=(MultiStreamGuard&& other) = delete;

 private:
  c10::impl::InlineMultiStreamGuard<c10::backend::impl::DeviceGuardImpl> guard_;

  static std::vector<c10::Stream> unwrapStreams(at::ArrayRef<Stream> streams) {
    std::vector<c10::Stream> streams_;
    streams_.reserve(streams.size());
    for (const Stream& stream : streams) {
      streams_.push_back(stream);
    }
    return streams_;
  }
};

} // namespace c10::backend
