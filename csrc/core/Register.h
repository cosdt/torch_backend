#pragma once

// Declared in c10/core/DeviceType.h
#define C10_REGISTER_PRIVATEUSE1_BACKEND(name)                            \
  int register_privateuse1_backend() {                                    \
    c10::register_privateuse1_backend(#name);                             \
    return 0;                                                             \
  }                                                                       \
  static const int _temp_backend_##name = register_privateuse1_backend();

// Declared in c10/core/StorageImpl.h
#define C10_SET_STORAGE_IMPL_CREATE(make_storage_impl)                           \
  int set_storage_impl_create() {                                                \
    c10::SetStorageImplCreate(c10::DeviceType::PrivateUse1, make_storage_impl);  \
    return 0;                                                                    \
  }                                                                              \
  static const int _temp_storage_impl = set_storage_impl_create();

// Declared in ATen/detail/PrivateUse1HooksInterface.h
#define AT_REGISTER_PRIVATEUSE1_HOOKS_INTERFACE(get_hooks)            \
  int register_privateuse1_hooks_interface() {                        \
    at::RegisterPrivateUse1HooksInterface(get_hooks);                 \
    return 0;                                                         \
  }                                                                   \
  static const int _temp_hooks = register_privateuse1_hooks_interface();

// Declared in torch/csrc/jit/serialization/pickler.h
#define REGISTER_TENSOR_BACKEND_META_REGISTRY(serialization, deserialization)  \
  int register_tensor_backend_meta_registry() {                                \
    torch::jit::TensorBackendMetaRegistry(                                     \
        c10::DeviceType::PrivateUse1,                                          \
        serialization,                                                         \
        deserialization);                                                      \
    return 0;                                                                  \
  }                                                                            \
  static const int _temp_tensor_meta = register_tensor_backend_meta_registry();
