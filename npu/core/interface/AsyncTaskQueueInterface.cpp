#include "AsyncTaskQueueInterface.h"
#include <ATen/record_function.h>
#include "npu/acl/include/acl/acl_rt.h"
#include "npu/core/register/OptionsManager.h"

namespace c10::npu {
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
  c10::npu::NPUStream stream = c10::npu::getCurrentNPUStream();
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

} // namespace queue
} // namespace c10::npu
