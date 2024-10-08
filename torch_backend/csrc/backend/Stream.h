#pragma once

#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>
#include "csrc/backend/Stream.h"
#include "csrc/core/Macros.h"

namespace torch::backend::stream {

struct THNPStream : THPStream {
  c10::backend::Stream npu_stream;
};

TORCH_BACKEND_API void init(PyObject* module);

TORCH_BACKEND_API PyMethodDef* python_functions();

} // namespace torch::backend::stream
