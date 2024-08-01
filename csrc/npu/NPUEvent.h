#pragma once

#include <cstdint>
#include <utility>
#include "csrc/core/Macros.h"
#include "csrc/npu/NPUStream.h"
#include "npu/acl/include/acl/acl.h"
#include "npu/core/NPUException.h"
#include "npu/core/NPUGuard.h"
#include "npu/core/sys_ctrl/npu_sys_ctrl.h"

namespace c10_npu {
/*
 * NPUEvents are movable not copyable wrappers around NPU's events.
 * NPUEvents are constructed lazily when first recorded.
 */
struct C10_BACKEND_API NPUEvent {
  // Constructors
  // Default value for `flags` is specified below
  NPUEvent() noexcept = default;
  NPUEvent(unsigned int flags) noexcept : flags_{flags} {}

  ~NPUEvent() {
    try {
      if (is_created_ && c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
        NPUGuard guard(device_index_);
        NPU_CHECK_ERROR(aclrtDestroyEvent(event_));
      }
    } catch (...) { /* No throw */ }
  }

  NPUEvent(const NPUEvent&) = delete;
  NPUEvent& operator=(const NPUEvent&) = delete;

  NPUEvent(NPUEvent&& other) noexcept { moveHelper(std::move(other)); }
  NPUEvent& operator=(NPUEvent&& other) noexcept {
    if (this != &other) {
      moveHelper(std::move(other));
    }
    return *this;
  }

  operator aclrtEvent() const { return event(); }

  // aclrtEvent do not support Less than operator until now

  c10::optional<at::Device> device() const {
    if (is_created_) {
      return at::Device(c10::DeviceType::PrivateUse1, device_index_);
    } else {
      return {};
    }
  }

  bool isCreated() const { return is_created_; }
  c10::DeviceIndex device_index() const { return device_index_; }
  aclrtEvent event() const { return event_; }

  bool query() const {
    if (!is_created_) {
      return true;
    }
    acl::aclrtEventRecordedStatus currStatus =
        acl::ACL_EVENT_RECORDED_STATUS_NOT_READY;
    NPU_CHECK_ERROR(acl::AclQueryEventRecordedStatus(event_, &currStatus));

    if (currStatus == acl::ACL_EVENT_RECORDED_STATUS_COMPLETE) {
      return true;
    }
    return false;
  }

  void record() { record(getCurrentNPUStream()); }

  void recordOnce(const NPUStream& stream) {
    if (!was_recorded_) record(stream);
  }

  void record(const NPUStream& stream) {
    if (!is_created_) {
      createEvent(stream.device_index());
    }

    TORCH_CHECK(
        device_index_ == stream.device_index(),
        "Event device ",
        device_index_,
        " does not match recording stream's device ",
        stream.device_index(),
        ".",
        PTA_ERROR(ErrCode::PARAM));
    NPUGuard guard(device_index_);
    NPU_CHECK_ERROR(aclrtRecordEvent(event_, stream));
    was_recorded_ = true;
  }

  void block(const NPUStream& stream) {
    if (is_created_) {
      NPUGuard guard(stream.device_index());
      NPU_CHECK_ERROR(aclrtStreamWaitEvent(stream, event_));
    }
  }

  float elapsed_time(const NPUEvent& other) const {
    TORCH_CHECK(is_created_ && other.isCreated(),
        "Both events must be recorded before calculating elapsed time.",
        PTA_ERROR(ErrCode::INTERNAL));
    float time_ms = 0;
    // We do not strictly have to set the device index to the same as our event,
    // but if we don't and the current device is not initialized, it will
    // create a new NPU context, which will consume a lot of memory.
    NPUGuard guard(device_index_);
    // raise error if either event is recorded but not yet completed
    NPU_CHECK_ERROR(aclrtEventElapsedTime(&time_ms, event_, other.event_));
    return time_ms;
  }

  void synchronize() const {
    if (is_created_) {
      NPU_CHECK_ERROR(aclrtSynchronizeEvent(event_));
      ASCEND_LOGI(
          "Event: aclrtSynchronizeEvent is successfully executed, event=%p",
          event_);
    }
  }

  // npu do not support IpcEventHandle until now

 private:
  unsigned int flags_ = ACL_EVENT_SYNC;
  bool is_created_ = false;
  bool was_recorded_ = false;
  c10::DeviceIndex device_index_ = -1;
  aclrtEvent event_ = nullptr;

  void createEvent(c10::DeviceIndex device_index) {
    device_index_ = device_index;
    NPUGuard guard(device_index_);
    NPU_CHECK_ERROR(c10_npu::acl::AclrtCreateEventWithFlag(&event_, flags_));
    ASCEND_LOGI(
        "Event: aclrtCreateEventWithFlag is successfully executed, event=%p",
        event_);
    is_created_ = true;
  }

  void moveHelper(NPUEvent&& other) {
    std::swap(flags_, other.flags_);
    std::swap(is_created_, other.is_created_);
    std::swap(was_recorded_, other.was_recorded_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
  }
};

} // namespace c10_npu
