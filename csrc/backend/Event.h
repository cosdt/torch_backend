#pragma once

#include <cstdint>
#include <utility>
#include "csrc/backend/DeviceGuard.h"
#include "csrc/backend/Stream.h"
#include "csrc/core/Macros.h"

// TODO(FFFrog):
// Remove later
#include "acl/include/acl/acl.h"
#include "core/NPUException.h"

namespace c10::backend {
/*
 * Events are movable not copyable wrappers around device's events.
 * Events are constructed lazily when first recorded.
 */
struct C10_BACKEND_API Event {
  // Constructors
  // Default value for `flags` is specified below
  Event() noexcept = default;
  Event(unsigned int flags) noexcept : flags_{flags} {}

  ~Event() {
    try {
      if (is_created_) {
        DeviceGuard guard(device_index_);
        aclrtDestroyEvent(event_);
      }
    } catch (...) { /* No throw */
    }
  }

  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

  Event(Event&& other) noexcept {
    moveHelper(std::move(other));
  }
  Event& operator=(Event&& other) noexcept {
    if (this != &other) {
      moveHelper(std::move(other));
    }
    return *this;
  }

  operator aclrtEvent() const {
    return event();
  }

  // aclrtEvent do not support Less than operator until now

  c10::optional<at::Device> device() const {
    if (is_created_) {
      return at::Device(c10::DeviceType::PrivateUse1, device_index_);
    } else {
      return {};
    }
  }

  bool isCreated() const {
    return is_created_;
  }
  c10::DeviceIndex device_index() const {
    return device_index_;
  }
  aclrtEvent event() const {
    return event_;
  }

  bool query() const {
    if (!is_created_) {
      return true;
    }
    aclrtEventRecordedStatus currStatus = ACL_EVENT_RECORDED_STATUS_NOT_READY;
    aclrtQueryEventStatus(event_, &currStatus);

    if (currStatus == ACL_EVENT_RECORDED_STATUS_COMPLETE) {
      return true;
    }
    return false;
  }

  void record() {
    record(getCurrentStream());
  }

  void recordOnce(const Stream& stream) {
    if (!was_recorded_)
      record(stream);
  }

  void record(const Stream& stream) {
    if (!is_created_) {
      createEvent(stream.device_index());
    }

    TORCH_CHECK(
        device_index_ == stream.device_index(),
        "Event device ",
        device_index_,
        " does not match recording stream's device ",
        stream.device_index(),
        ".");
    DeviceGuard guard(device_index_);
    aclrtRecordEvent(event_, stream);
    was_recorded_ = true;
  }

  void block(const Stream& stream) {
    if (is_created_) {
      DeviceGuard guard(stream.device_index());
      aclrtStreamWaitEvent(stream, event_);
    }
  }

  float elapsed_time(const Event& other) const {
    TORCH_CHECK(
        is_created_ && other.isCreated(),
        "Both events must be recorded before calculating elapsed time.");
    float time_ms = 0;
    // We do not strictly have to set the device index to the same as our event,
    // but if we don't and the current device is not initialized, it will
    // create a new context, which will consume a lot of memory.
    DeviceGuard guard(device_index_);
    // raise error if either event is recorded but not yet completed
    aclrtEventElapsedTime(&time_ms, event_, other.event_);
    return time_ms;
  }

  void synchronize() const {
    if (is_created_) {
      aclrtSynchronizeEvent(event_);
    }
  }

  // do not support IpcEventHandle until now

 private:
  unsigned int flags_ = ACL_EVENT_SYNC;
  bool is_created_ = false;
  bool was_recorded_ = false;
  c10::DeviceIndex device_index_ = -1;
  aclrtEvent event_ = nullptr;

  void createEvent(c10::DeviceIndex device_index) {
    device_index_ = device_index;
    DeviceGuard guard(device_index_);
    aclrtCreateEventExWithFlag(&event_, flags_);
    is_created_ = true;
  }

  void moveHelper(Event&& other) {
    std::swap(flags_, other.flags_);
    std::swap(is_created_, other.is_created_);
    std::swap(was_recorded_, other.was_recorded_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
  }
};

} // namespace c10::backend
