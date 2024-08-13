#include <torch/csrc/utils/tensor_new.h>

#include "csrc/backend/Functions.h"
#include "csrc/core/Macros.h"

namespace torch::backend::tensor {

TORCH_BACKEND_API PyMethodDef* python_functions();

} // namespace torch::backend::tensor
