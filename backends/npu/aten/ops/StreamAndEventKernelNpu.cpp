#include "csrc/aten/generated/NPUNativeFunctions.h"
#include "framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

void NPUNativeFunctions::record_stream(at::Tensor& self, c10::Stream stream) {
  struct c10::StreamData3 data = stream.pack3();
  c10::backend::Allocator::recordStream(
      self.storage().data_ptr(),
      c10::backend::Stream::unpack3(
          data.stream_id, data.device_index, data.device_type));
}

} // namespace native
} // namespace at_npu
