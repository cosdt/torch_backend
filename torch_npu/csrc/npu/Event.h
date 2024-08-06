#pragma once

#include <torch/csrc/python_headers.h>
#include "csrc/core/Macros.h"
#include "csrc/npu/NPUEvent.h"

namespace torch::backend::event {

struct THNPEvent {
  PyObject_HEAD c10::npu::NPUEvent npu_event;
};
extern PyObject* THNPEventClass;

TORCH_BACKEND_API void init(PyObject* module);

} // namespace torch::backend::event
