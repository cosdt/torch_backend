#pragma once

#include <Python.h>
#include "csrc/core/Macros.h"

namespace torch::backend::init {

TORCH_BACKEND_API PyMethodDef* python_functions();

} // namespace torch::backend::init
