#include "AsyncTaskQueueInterface.h"
#include <ATen/record_function.h>
#include "csrc/npu/NPUEventManager.h"
#include "npu/core/register/OptionsManager.h"
#include "npu/framework/NPUDefine.h"
#include "npu/framework/utils/NpuUtils.h"
#include "npu/acl/include/acl/acl_rt.h"

namespace c10_npu {
namespace queue {
std::atomic<uint64_t> QueueParas::g_correlation_id{0};
std::map<int64_t, std::string> CopyParas::COPY_PARAS_MAP{
    {ACL_MEMCPY_HOST_TO_HOST, "acl_memcpy_host_to_host"},
    {ACL_MEMCPY_HOST_TO_DEVICE, "acl_memcpy_host_to_device"},
    {ACL_MEMCPY_DEVICE_TO_HOST, "acl_memcpy_device_to_host"},
    {ACL_MEMCPY_DEVICE_TO_DEVICE, "acl_memcpy_device_to_device"},
};
std::map<int64_t, std::string> EventParas::EVENT_PARAS_MAP{
    {RECORD_EVENT, "record_event"},
    {WAIT_EVENT, "wait_event"},
    {LAZY_DESTROY_EVENT, "destroy_event"},
};
void CopyParas::Copy(CopyParas& other) {
  this->dst = other.dst;
  this->dstLen = other.dstLen;
  this->src = other.src;
  this->srcLen = other.srcLen;
  this->kind = other.kind;
}

void EventParas::Copy(EventParas& other) {
  this->event = other.event;
  this->eventAllocatorType = other.eventAllocatorType;
}

class AsyncCopyTask {
 public:
  AsyncCopyTask(
      void* dst,
      size_t dstLen,
      void* src,
      size_t srcLen,
      aclrtMemcpyKind kind);
  ~AsyncCopyTask() = default;
  void LaunchCopyTask();

 private:
  CopyParas copyParam_;
};

class EventTask {
 public:
  explicit EventTask(
      aclrtEvent event,
      EventAllocatorType allocatorType = RESERVED)
      : eventParam_(event, allocatorType) {};
  ~EventTask() = default;
  void LaunchRecordTask(c10_npu::NPUStream npuStream);
  void LaunchWaitTask(c10_npu::NPUStream npuStream);
  void LaunchLazyDestroyTask(c10::DeviceIndex device_index);

 private:
  EventParas eventParam_;
};

AsyncCopyTask::AsyncCopyTask(
    void* dst,
    size_t dstLen,
    void* src,
    size_t srcLen,
    aclrtMemcpyKind kind) {
  copyParam_.dst = dst;
  copyParam_.dstLen = dstLen;
  copyParam_.src = src;
  copyParam_.srcLen = srcLen;
  copyParam_.kind = kind;
}

void AsyncCopyTask::LaunchCopyTask() {
  RECORD_FUNCTION(
      CopyParas::COPY_PARAS_MAP[copyParam_.kind], std::vector<c10::IValue>({}));
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
    NPU_CHECK_ERROR(aclrtMemcpyAsync(
        copyParam_.dst,
        copyParam_.dstLen,
        copyParam_.src,
        copyParam_.srcLen,
        copyParam_.kind,
        stream));
}

aclError LaunchAsyncCopyTask(
    void* dst,
    size_t dstLen,
    void* src,
    size_t srcLen,
    aclrtMemcpyKind kind) {
  AsyncCopyTask copyTask(dst, dstLen, src, srcLen, kind);
  copyTask.LaunchCopyTask();
  return ACL_ERROR_NONE;
}

void EventTask::LaunchRecordTask(c10_npu::NPUStream npuStream) {
  RECORD_FUNCTION(
      EventParas::EVENT_PARAS_MAP[RECORD_EVENT], std::vector<c10::IValue>({}));
    NPU_CHECK_ERROR(aclrtRecordEvent(eventParam_.event, npuStream));
    ASCEND_LOGI(
        "Event: aclrtRecordEvent is successfully executed, stream=%p, event=%p",
        npuStream.stream(false),
        eventParam_.event);
}

aclError LaunchRecordEventTask(aclrtEvent event, c10_npu::NPUStream npuStream) {
  EventTask recordTask(event);
  recordTask.LaunchRecordTask(npuStream);
  return ACL_ERROR_NONE;
}

void EventTask::LaunchWaitTask(c10_npu::NPUStream npuStream) {
  RECORD_FUNCTION(
      EventParas::EVENT_PARAS_MAP[WAIT_EVENT], std::vector<c10::IValue>({}));
    NPU_CHECK_ERROR(aclrtStreamWaitEvent(npuStream, eventParam_.event));
    ASCEND_LOGI(
        "Event: aclrtStreamWaitEvent is successfully executed, stream=%p, event=%p",
        npuStream.stream(false),
        eventParam_.event);
}

aclError LaunchWaitEventTask(aclrtEvent event, c10_npu::NPUStream npuStream) {
  EventTask waitTask(event);
  waitTask.LaunchWaitTask(npuStream);
  return ACL_ERROR_NONE;
}

void EventTask::LaunchLazyDestroyTask(c10::DeviceIndex device_index) {
  RECORD_FUNCTION(
      EventParas::EVENT_PARAS_MAP[LAZY_DESTROY_EVENT],
      std::vector<c10::IValue>({}));
    NPU_CHECK_ERROR(
        c10_npu::NPUEventManager::GetInstance().LazyDestroy(eventParam_.event),
        "aclrtDestroyEvent");
}

aclError LaunchLazyDestroyEventTask(
    aclrtEvent event,
    c10::DeviceIndex device_index) {
  EventTask lazyDestroyTask(event);
  lazyDestroyTask.LaunchLazyDestroyTask(device_index);
  return ACL_ERROR_NONE;
}
} // namespace queue
} // namespace c10_npu
