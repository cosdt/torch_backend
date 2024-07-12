#include "csrc/npu/NPUEvent.h"
#include "csrc/npu/NPUEventManager.h"
#include "npu/core/NPUException.h"
#include "npu/core/NPUGuard.h"
#include "npu/core/interface/AsyncTaskQueueInterface.h"
#include "npu/core/register/OptionsManager.h"
#include "npu/core/sys_ctrl/npu_sys_ctrl.h"

namespace c10_npu {
unsigned int NPUEvent::defaultFlags() {
  return c10_npu::acl::IsExistCreateEventExWithFlag() ? ACL_EVENT_SYNC
                                                      : ACL_EVENT_DEFAULT;
}

NPUEvent::~NPUEvent() {
  try {
    if (is_created_ && (c10_npu::NpuSysCtrl::GetInstance().GetInitFlag())) {
      NPU_CHECK_ERROR(
          c10_npu::queue::LaunchLazyDestroyEventTask(event_, device_index_));
      if (!c10_npu::acl::IsExistCreateEventExWithFlag()) {
        c10_npu::NPUEventManager::GetInstance().QueryAndDestroyEvent();
      }
    }
  } catch (...) {
    // stay consistent with pytorch, no throw
  }
}

bool NPUEvent::query() const {
  if (!is_created_) {
    return true;
  }
  if (c10_npu::option::OptionsManager::CheckQueueEnable() &&
      !c10_npu::NPUEventManager::GetInstance().IsEventRecorded(event_)) {
    return false;
  }
  acl::aclrtEventRecordedStatus currStatus =
      acl::ACL_EVENT_RECORDED_STATUS_NOT_READY;
  NPU_CHECK_ERROR(acl::AclQueryEventRecordedStatus(event_, &currStatus));

  if (currStatus == acl::ACL_EVENT_RECORDED_STATUS_COMPLETE) {
    return true;
  }
  return false;
}

void NPUEvent::record(const NPUStream& stream) {
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
  NPU_CHECK_ERROR(c10_npu::queue::LaunchRecordEventTask(event_, stream));
  was_recorded_ = true;
}

void NPUEvent::block(const NPUStream& stream) {
  if (is_created_) {
    NPUGuard guard(stream.device_index());
    NPU_CHECK_ERROR(c10_npu::queue::LaunchWaitEventTask(event_, stream));
  }
}

float NPUEvent::elapsed_time(const NPUEvent& other) const {
  TORCH_CHECK(
      is_created_ && other.isCreated(),
      "Both events must be recorded before calculating elapsed time.",
      PTA_ERROR(ErrCode::INTERNAL));
  float time_ms = 0;
  NPUStatus ret = c10_npu::emptyAllNPUStream();
  if (ret != SUCCESS) {
    ASCEND_LOGE("MakeSureQueueEmpty fail, ret: %s", ret.c_str());
  }
  NPU_CHECK_ERROR(aclrtSynchronizeEvent(event_));
  ASCEND_LOGI(
      "Event: aclrtSynchronizeEvent is successfully executed, event=%p",
      event_);
  NPU_CHECK_ERROR(aclrtSynchronizeEvent(other.event_));
  ASCEND_LOGI(
      "Event: aclrtSynchronizeEvent is successfully executed, other.event=%p",
      other.event_);
  // raise error if either event is recorded but not yet completed
  NPU_CHECK_ERROR(aclrtEventElapsedTime(&time_ms, event_, other.event_));
  return time_ms;
}

void NPUEvent::synchronize() const {
  if (is_created_) {
    NPUStatus ret = c10_npu::emptyAllNPUStream();
    if (ret != SUCCESS) {
      ASCEND_LOGE("MakeSureQueueEmpty fail, ret: %s", ret.c_str());
    }
    NPU_CHECK_ERROR(aclrtSynchronizeEvent(event_));
    ASCEND_LOGI(
        "Event: aclrtSynchronizeEvent is successfully executed, event=%p",
        event_);
  }
}

void NPUEvent::createEvent(c10::DeviceIndex device_index) {
  device_index_ = device_index;
  NPUGuard guard(device_index_);
  NPU_CHECK_ERROR(c10_npu::acl::AclrtCreateEventWithFlag(&event_, flags_));
  ASCEND_LOGI(
      "Event: aclrtCreateEventWithFlag is successfully executed, event=%p",
      event_);
  is_created_ = true;
}
} // namespace c10_npu
