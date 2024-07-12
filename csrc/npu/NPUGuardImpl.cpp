#pragma GCC visibility push(default)
#include "csrc/npu/NPUGuardImpl.h"
#include <torch/csrc/jit/serialization/pickler.h>
#include "csrc/npu/NPUSerialization.h"
#include "csrc/npu/NPUStorageImpl.h"
#include "csrc/npu/NPUHooksInterface.h"

namespace c10_npu {

namespace impl {

C10_REGISTER_GUARD_IMPL(PrivateUse1, NPUGuardImpl);

#define REGISTER_PRIVATEUSE1_BACKEND(name)                                \
  int rename_privateuse1_backend() {                                      \
    c10::register_privateuse1_backend(#name);                             \
    c10::SetStorageImplCreate(                                            \
        c10::DeviceType::PrivateUse1, &torch_npu::make_npu_storage_impl); \
    at::RegisterPrivateUse1HooksInterface(c10_npu::get_npu_hooks());      \
    torch::jit::TensorBackendMetaRegistry(                                \
        c10::DeviceType::PrivateUse1,                                     \
        &torch_npu::npu_info_serialization,                               \
        &torch_npu::npu_info_deserialization);                            \
    return 0;                                                             \
  }                                                                       \
  static const int _temp_##name = rename_privateuse1_backend();

REGISTER_PRIVATEUSE1_BACKEND(npu)

} // namespace impl

} // namespace c10_npu
#pragma GCC visibility pop
