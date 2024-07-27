#include <torch/csrc/utils/python_stub.h>
#include "csrc/core/Macros.h"
#include "npu/core/NPUException.h"

namespace torch_npu {
namespace autocast {

TORCH_BACKEND_API PyMethodDef* autocast_mode_functions();

}
} // namespace torch_npu
