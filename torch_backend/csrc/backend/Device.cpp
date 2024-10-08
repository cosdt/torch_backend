#include <c10/core/DeviceType.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>

#include "csrc/backend/Context.h"
#include "csrc/backend/DeviceCachingAllocator.h"
#include "csrc/backend/DeviceGuard.h"
#include "csrc/backend/DeviceProp.h"
#include "csrc/backend/Functions.h"

#include "torch_backend/csrc/backend/Device.h"

#define CHANGE_UNIT_SIZE 1024.0

namespace torch::backend::device {

void registerDeviceProperties(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<c10::backend::DeviceProp>(m, "_NPUDeviceProperties")
      .def_readonly("name", &c10::backend::DeviceProp::name)
      .def_readonly("total_memory", &c10::backend::DeviceProp::totalGlobalMem)
      .def("__repr__", [](const c10::backend::DeviceProp& prop) {
        std::ostringstream stream;
        stream << "_NPUDeviceProperties(name='" << prop.name
               << "', total_memory="
               << prop.totalGlobalMem / (CHANGE_UNIT_SIZE * CHANGE_UNIT_SIZE)
               << "MB)";
        return stream.str();
      });

  m.def("_npu_isHistoryEnabled", []() {
    return c10::backend::Allocator::isHistoryEnabled();
  });
}

void bindGetDeviceProperties(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def(
      "_npu_getDeviceProperties",
      [](c10::DeviceIndex deviceid) -> c10::backend::DeviceProp* {
        return c10::backend::getDeviceProperties(deviceid);
      },
      py::return_value_policy::reference);
}

void init(PyObject* module) {
  registerDeviceProperties(module);
  bindGetDeviceProperties(module);
}

PyObject* THPModule_npuSynchronize(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  pybind11::gil_scoped_release no_gil;
  c10::backend::device_synchronize();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_setDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to setDevice");
  {
    pybind11::gil_scoped_release no_gil;
    at::globalContext().lazyInitPrivateUse1();
  }

  auto device = THPUtils_unpackLong(arg);
  c10::backend::set_device(static_cast<c10::DeviceIndex>(device));

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_getDevice_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::device_lazy_init(at::kPrivateUse1);
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  auto device = static_cast<int32_t>(c10::backend::current_device());
  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_getDeviceCount_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(c10::backend::device_count());
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_npuCanDeviceAccessPeer_wrap(
    PyObject* self,
    PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* value_1 = nullptr;
  PyObject* value_2 = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &value_1, &value_2)) {
    throw torch::TypeError("Pybind failed to parse parameters.");
  }
  c10::DeviceIndex device_id = THPUtils_unpackDeviceIndex(value_1);
  c10::DeviceIndex peer_device_id = THPUtils_unpackDeviceIndex(value_2);
  int32_t can_access_peer = 0;
  NPU_CHECK_ERROR(
      aclrtDeviceCanAccessPeer(&can_access_peer, device_id, peer_device_id));
  auto can_device_access_peer = can_access_peer != 0;
  return PyBool_FromLong(can_device_access_peer);
  END_HANDLE_TH_ERRORS
}

static struct PyMethodDef THPModule_methods[] = {
    {"_synchronize",
     (PyCFunction)THPModule_npuSynchronize,
     METH_NOARGS,
     nullptr},
    {"_setDevice", (PyCFunction)THPModule_setDevice_wrap, METH_O, nullptr},
    {"_getDevice", (PyCFunction)THPModule_getDevice_wrap, METH_NOARGS, nullptr},
    {"_getDeviceCount",
     (PyCFunction)THPModule_getDeviceCount_wrap,
     METH_NOARGS,
     nullptr},
    {"_canDeviceAccessPeer",
     (PyCFunction)THPModule_npuCanDeviceAccessPeer_wrap,
     METH_VARARGS,
     nullptr},
    {nullptr}};

PyMethodDef* python_functions() {
  return THPModule_methods;
}

} // namespace torch::backend::device
