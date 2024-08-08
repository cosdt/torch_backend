#include "torch_backend/csrc/backend/Stream.h"
#include <pybind11/pybind11.h>
#include <structmember.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include "csrc/backend/NPUGuard.h"

namespace torch::backend::stream {

PyObject* THNPStreamClass = nullptr;

static PyObject* THNPStream_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS

  c10::DeviceIndex current_device;
  NPU_CHECK_ERROR(c10::backend::GetDevice(&current_device));

  int priority = 0;
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;
  uint64_t stream_ptr = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "priority",
      "stream_id",
      "device_index",
      "device_type",
      "stream_ptr",
      nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|iLLLK",
          const_cast<char**>(kwlist),
          &priority,
          &stream_id,
          &device_index,
          &device_type,
          &stream_ptr)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  c10::backend::NPUStream stream = (stream_id || device_index || device_type)
      ? c10::backend::NPUStream::unpack3(
            stream_id, device_index, static_cast<c10::DeviceType>(device_type))
      : c10::backend::getStreamFromPool(false);

  THNPStream* self = (THNPStream*)ptr.get();
  self->stream_id = static_cast<int64_t>(stream.id());
  self->device_index = static_cast<int64_t>(stream.device_index());
  self->device_type = static_cast<int64_t>(stream.device_type());
  new (&self->npu_stream) c10::backend::NPUStream(stream);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THNPStream_dealloc(THNPStream* self) {
  self->npu_stream.~NPUStream();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THNPStream_get_device(THNPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(self->npu_stream.device());
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_get_npu_stream(THNPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(self->npu_stream.stream());
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_get_priority(THNPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(false, "NPU dose not support Stream.get_priority() currently.");
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_priority_range() {
  HANDLE_TH_ERRORS
  TORCH_CHECK(false, "NPU does not support Stream.priority_range() currently.");
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_query(THNPStream* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->npu_stream.query());
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_synchronize(THNPStream* self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil;
    self->npu_stream.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_eq(THNPStream* self, THNPStream* other) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->npu_stream == other->npu_stream);
  END_HANDLE_TH_ERRORS
}

static struct PyMemberDef THNPStream_members[] = {
    {(char*)"stream_id",
     T_ULONGLONG,
     offsetof(THNPStream, stream_id),
     READONLY,
     nullptr},
    {(char*)"device_type",
     T_ULONGLONG,
     offsetof(THNPStream, device_type),
     READONLY,
     nullptr},
    {(char*)"device_index",
     T_ULONGLONG,
     offsetof(THNPStream, device_index),
     READONLY,
     nullptr},
    {nullptr}};

static struct PyGetSetDef THNPStream_properties[] = {
    {"device", (getter)THNPStream_get_device, nullptr, nullptr, nullptr},
    {"npu_stream",
     (getter)THNPStream_get_npu_stream,
     nullptr,
     nullptr,
     nullptr},
    {"priority", (getter)THNPStream_get_priority, nullptr, nullptr, nullptr},
    {nullptr}};

static PyMethodDef THNPStream_methods[] = {
    {(char*)"query", (PyCFunction)THNPStream_query, METH_NOARGS, nullptr},
    {(char*)"synchronize",
     (PyCFunction)THNPStream_synchronize,
     METH_NOARGS,
     nullptr},
    {(char*)"priority_range",
     (PyCFunction)(void (*)(void))THNPStream_priority_range,
     METH_STATIC | METH_NOARGS,
     nullptr},
    {(char*)"__eq__", (PyCFunction)THNPStream_eq, METH_O, nullptr},
    {nullptr}};

PyTypeObject THNPStreamType = {
    PyVarObject_HEAD_INIT(
        nullptr,
        0) "torch_backend._C._NPUStreamBase", /* tp_name */
    sizeof(THNPStream), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THNPStream_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    0, /* tp_getattr */
    0, /* tp_setattr */
    0, /* tp_reserved */
    0, /* tp_repr */
    0, /* tp_as_number */
    0, /* tp_as_sequence */
    0, /* tp_as_mapping */
    0, /* tp_hash  */
    0, /* tp_call */
    0, /* tp_str */
    0, /* tp_getattro */
    0, /* tp_setattro */
    0, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    THNPStream_methods, /* tp_methods */
    THNPStream_members, /* tp_members */
    THNPStream_properties, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    0, /* tp_init */
    0, /* tp_alloc */
    THNPStream_pynew, /* tp_new */
};

void init(PyObject* module) {
  Py_INCREF(THPStreamClass);
  THNPStreamType.tp_base = THPStreamClass;
  THNPStreamClass = (PyObject*)&THNPStreamType;
  if (PyType_Ready(&THNPStreamType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THNPStreamType);
  if (PyModule_AddObject(module, "_NPUStreamBase", (PyObject*)&THNPStreamType) <
      0) {
    throw python_error();
  }
}

std::vector<c10::optional<c10::backend::NPUStream>>
THNPUtils_PySequence_to_NPUStreamList(PyObject* obj) {
  if (!PySequence_Check(obj)) {
    throw std::runtime_error(
        "Expected a sequence in THNPUtils_PySequence_to_NPUStreamList");
  }
  THPObjectPtr seq = THPObjectPtr(PySequence_Fast(obj, nullptr));
  if (seq.get() == nullptr) {
    throw std::runtime_error(
        "expected PySequence, but got " + std::string(THPUtils_typename(obj)));
  }

  std::vector<c10::optional<c10::backend::NPUStream>> streams;
  Py_ssize_t length = PySequence_Fast_GET_SIZE(seq.get());
  for (Py_ssize_t i = 0; i < length; i++) {
    PyObject* stream = PySequence_Fast_GET_ITEM(seq.get(), i);

    if (PyObject_IsInstance(stream, THNPStreamClass)) {
      streams.emplace_back(c10::backend::NPUStream::unpack3(
          (reinterpret_cast<THNPStream*>(stream))->stream_id,
          (reinterpret_cast<THNPStream*>(stream))->device_index,
          static_cast<c10::DeviceType>(
              (reinterpret_cast<THNPStream*>(stream))->device_type)));
    } else if (stream == Py_None) {
      streams.emplace_back();
    } else {
      std::runtime_error(
          "Unknown data type found in stream list. Need torch_backend.npu.Stream or None");
    }
  }
  return streams;
}

PyObject* THNPModule_getCurrentStream_wrap(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");

  c10::DeviceIndex device = THPUtils_unpackDeviceIndex(device_index);
  auto stream = c10::backend::getCurrentNPUStream(device);
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(
      output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(
      output_tuple,
      1,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_index())));
  PyTuple_SetItem(
      output_tuple,
      2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_getDefaultStream_wrap(
    PyObject* self /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(device_index), "invalid argument to getDefaultStream");

  c10::DeviceIndex device = THPUtils_unpackDeviceIndex(device_index);
  auto stream = c10::backend::getDefaultNPUStream(device);
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(
      output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(
      output_tuple,
      1,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_index())));
  PyTuple_SetItem(
      output_tuple,
      2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_setStream_wrap(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "stream_id", "device_index", "device_type", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|LLL",
          const_cast<char**>(kwlist),
          &stream_id,
          &device_index,
          &device_type)) {
  }

  auto stream = c10::backend::NPUStream::unpack3(
      stream_id,
      static_cast<c10::DeviceIndex>(device_index),
      static_cast<c10::DeviceType>(device_type));

  auto device = c10::backend::current_device();
  if (device != stream.device_index()) {
    c10::backend::set_device(stream.device_index());
  }
  c10::backend::setCurrentNPUStream(stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static struct PyMethodDef THNPModule_methods[] = {
    {"_npu_getCurrentStream",
     (PyCFunction)THNPModule_getCurrentStream_wrap,
     METH_O,
     nullptr},
    {"_npu_getDefaultStream",
     (PyCFunction)THNPModule_getDefaultStream_wrap,
     METH_O,
     nullptr},
    {"_npu_setStream",
     (PyCFunction)THNPModule_setStream_wrap,
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {nullptr}};

PyMethodDef* python_functions() {
  return THNPModule_methods;
}

} // namespace torch::backend::stream
