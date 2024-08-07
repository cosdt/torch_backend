#include "GeneratorImpl.h"
#include <ATen/EmptyTensor.h>
#include <ATen/Utils.h>

namespace c10::backend::Generator {

GeneratorImpl::GeneratorImpl(DeviceIndex device_index)
    : c10::GeneratorImpl{
          Device(DeviceType::PrivateUse1, device_index),
          DispatchKeySet(DispatchKey::PrivateUse1)} {}

void GeneratorImpl::set_offset(uint64_t offset) {
  philox_offset_per_thread_ = offset;
}

uint64_t GeneratorImpl::get_offset() const {
  return philox_offset_per_thread_;
}

uint64_t GeneratorImpl::current_seed() const {
  return seed_;
}

void GeneratorImpl::set_current_seed(uint64_t seed) {
  seed_ = seed;
  philox_offset_per_thread_ = 0;
}

uint64_t GeneratorImpl::seed() {
  auto random = c10::detail::getNonDeterministicRandom();
  this->set_current_seed(random);
  return random;
}
} // namespace c10::backend::Generator
