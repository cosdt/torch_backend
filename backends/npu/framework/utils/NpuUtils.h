#ifndef __PULGIN_NATIVE_NPU_UTILS_NUP_UTILS__
#define __PULGIN_NATIVE_NPU_UTILS_NUP_UTILS__

#include <ATen/ATen.h>
#include <stdint.h>
#include <string>
#include <vector>
#include "core/npu_log.h"

#include "acl/include/acl/acl.h"
#include "acl/include/ge/ge_error_codes.h"

#include "core/interface/AsyncTaskQueueInterface.h"
#include "csrc/backend/DeviceCachingAllocator.h"
#include "framework/interface/AclOpCompileInterface.h"

using std::string;
using std::vector;

namespace at_npu {
namespace native {

// smallvector max size
const int N = 32;
// npu tensor max size
const int SHAPE_SIZE = 8;
// HALF_MAX and HALF_MIN of NPU support
const int NPU_HALF_MAX = 65504;
const int NPU_HALF_MIN = -65504;
const int NPU_MAX_OP_EXEC_TRY_NUM = 2;

typedef enum CompileType {
  MEMORY_HOST_COMPILE_DEPENDENT = 1,
  MEMORY_HOST_COMPILE_INDEPENDENT = 2,
} CompileType;

class NpuUtils {
 public:
  static bool check_match(const at::Tensor* tensor);
  static at::Tensor format_contiguous(const at::Tensor& src);
  static at::Tensor format_contiguous_add_copy_optimize(const at::Tensor& src);
  static void RefreshFormat(const at::Tensor& tensor);
  static void format_fresh_view(at::Tensor& x, const at::Tensor& y);

  static bool check_5d_5d_match(const at::Tensor& tensor);
  static bool IsOomError(aclError ret, int index);
  static void check_1d(const at::Tensor& t, const char* arg, const char* fn);
};

const std::string AclDateTypeToString(aclDataType descDType);
const std::string AclFormatToString(aclFormat descFormat);

} // namespace native
} // namespace at_npu

#endif // __NATIVE_NPU_UTILS_NUP_UTILS__
