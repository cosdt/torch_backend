#include <torch/csrc/utils/tensor_new.h>

#include "csrc/core/Macros.h"
#include "csrc/npu/NPUFunctions.h"

namespace torch::backend::tensor {

TORCH_BACKEND_API PyMethodDef* python_functions();

} // namespace torch_npu
