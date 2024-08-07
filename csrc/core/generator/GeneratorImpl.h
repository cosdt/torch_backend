#pragma once

#include <ATen/core/Generator.h>

namespace c10::backend::Generator {
struct C10_API GeneratorImpl : public c10::GeneratorImpl {
  GeneratorImpl(DeviceIndex device_index);

  GeneratorImpl(const GeneratorImpl& other) = delete;
  GeneratorImpl(GeneratorImpl&& other) = delete;
  GeneratorImpl& operator=(const GeneratorImpl& other) = delete;

  ~GeneratorImpl() override = default;

  c10::intrusive_ptr<GeneratorImpl> clone() const;

  virtual void set_current_seed(uint64_t seed) override;
  virtual void set_offset(uint64_t offset) override;
  virtual uint64_t get_offset() const override;
  virtual uint64_t current_seed() const override;
  virtual uint64_t seed() override;

 private:
  uint64_t philox_offset_per_thread_;
  uint64_t seed_ = default_rng_seed_val;
};

} // namespace c10::backend::Generator
