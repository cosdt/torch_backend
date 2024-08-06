#pragma once

#include <Python.h>
#include "csrc/core/Macros.h"

namespace torch::backend::device {

TORCH_BACKEND_API PyMethodDef* python_functions();

}

TORCH_BACKEND_API void RegisterNPUDeviceProperties(PyObject* module);
TORCH_BACKEND_API void BindGetDeviceProperties(PyObject* module);
