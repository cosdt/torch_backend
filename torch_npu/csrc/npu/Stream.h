#pragma once

#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>
#include "csrc/core/Macros.h"
#include "csrc/npu/NPUStream.h"

struct THNPStream : THPStream {
  c10_npu::NPUStream npu_stream;
};
extern PyObject* THNPStreamClass;

TORCH_BACKEND_API void THNPStream_init(PyObject* module);

inline bool THNPStream_Check(PyObject* obj) {
  return THNPStreamClass && PyObject_IsInstance(obj, THNPStreamClass);
}

TORCH_BACKEND_API std::vector<c10::optional<c10_npu::NPUStream>>
THNPUtils_PySequence_to_NPUStreamList(PyObject* obj);

TORCH_BACKEND_API PyMethodDef* THNPModule_stream_methods();
