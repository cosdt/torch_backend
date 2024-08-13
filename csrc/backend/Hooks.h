#pragma once
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Storage.h>
#include "csrc/backend/GeneratorImpl.h"

namespace c10::backend {

struct TORCH_API Hooks : public at::PrivateUse1HooksInterface {
  virtual ~Hooks() = default;
  const at::Generator& getDefaultGenerator(
      c10::DeviceIndex device_index) override {
    static auto device_gen =
        c10::backend::detail::getDefaultGenerator(device_index);
    return device_gen;
  }
  void initPrivateUse1() const override;
  bool hasPrimaryContext(c10::DeviceIndex device_index) const override;
  void resizePrivateUse1Bytes(const c10::Storage& storage, size_t new_bytes)
      const override;
  bool isPinnedPtr(const void* data) const;
  at::Allocator* getPinnedMemoryAllocator() const override;
};

struct TORCH_API HooksArgs : public at::PrivateUse1HooksArgs {};

// register to PrivateUse1HooksInterface
at::PrivateUse1HooksInterface* get_device_hooks();
} // namespace c10::backend
