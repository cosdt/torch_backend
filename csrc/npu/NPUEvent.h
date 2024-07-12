#pragma once

#include <cstdint>
#include <utility>
#include "csrc/npu/NPUStream.h"
#include "npu/core/NPUMacros.h"

namespace c10_npu {
/*
 * NPUEvents are movable not copyable wrappers around NPU's events.
 * NPUEvents are constructed lazily when first recorded.
 */
struct C10_NPU_API NPUEvent {
  // Constructors
  // Default value for `flags` is specified below
  NPUEvent() : flags_(defaultFlags()) {}
  NPUEvent(unsigned int flags) : flags_(flags) {}
  ~NPUEvent();

  NPUEvent(const NPUEvent&) = delete;
  NPUEvent& operator=(const NPUEvent&) = delete;

  NPUEvent(NPUEvent&& other) {
    moveHelper(std::move(other));
  }
  NPUEvent& operator=(NPUEvent&& other) {
    moveHelper(std::move(other));
    return *this;
  }

  operator void*() const {
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
  void* event() const {
    return event_;
  }

  bool query() const;
  void record() {
    record(getCurrentNPUStream());
  }
  void recordOnce(const NPUStream& stream) {
    if (!was_recorded_)
      record(stream);
  }
  void record(const NPUStream& stream);
  void block(const NPUStream& stream);
  float elapsed_time(const NPUEvent& other) const;
  void synchronize() const;

  // npu do not support IpcEventHandle until now

 private:
  unsigned int flags_;
  bool is_created_ = false;
  bool was_recorded_ = false;
  c10::DeviceIndex device_index_ = -1;
  void* event_ = nullptr;

  void createEvent(c10::DeviceIndex device_index);
  unsigned int defaultFlags();
  void moveHelper(NPUEvent&& other) {
    flags_ = defaultFlags();
    std::swap(is_created_, other.is_created_);
    std::swap(was_recorded_, other.was_recorded_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
  }
};

} // namespace c10_npu
