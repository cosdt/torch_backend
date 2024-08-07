#pragma once

#include <Python.h>
#include "csrc/core/Macros.h"

namespace torch::backend::lock {

TORCH_BACKEND_API PyMethodDef* python_functions();

} // namespace torch::backend::lock
