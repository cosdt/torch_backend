#include <gtest/gtest.h>
#include "csrc/backend/Context.h"
#include "csrc/backend/GeneratorImpl.h"

TEST(DeviceGeneratorImpl, TestSingletonDefaultGenerator) {
  if (!c10::backend::is_available()) {
    GTEST_SKIP() << "NPU is not available";
  }

  auto gen = c10::backend::detail::getDefaultGenerator();
  auto other = c10::backend::detail::getDefaultGenerator();
  EXPECT_EQ(gen, other);
}

TEST(DeviceGeneratorImpl, TestCloning) {
  if (!c10::backend::is_available()) {
    GTEST_SKIP() << "NPU is not available";
  }

  auto gen1 = c10::backend::detail::createGenerator();
  auto npu_gen1 = at::check_generator<c10::backend::DeviceGeneratorImpl>(gen1);

  auto gen2 = c10::backend::detail::createGenerator();
  gen2 = gen1.clone();
  auto npu_gen2 = at::check_generator<c10::backend::DeviceGeneratorImpl>(gen2);

  EXPECT_EQ(npu_gen1->current_seed(), npu_gen2->current_seed());
  EXPECT_EQ(
      npu_gen1->philox_offset_per_thread(),
      npu_gen2->philox_offset_per_thread());
}

TEST(DeviceGeneratorImpl, TestGetSetCurrentSeed) {
  if (!c10::backend::is_available()) {
    GTEST_SKIP() << "NPU is not available";
  }

  auto gen = c10::backend::detail::createGenerator();
  auto npu_gen = at::check_generator<c10::backend::DeviceGeneratorImpl>(gen);
  npu_gen->set_current_seed(10);
  EXPECT_EQ(npu_gen->current_seed(), 10);
}

TEST(DeviceGeneratorImpl, TestDeviceType) {
  EXPECT_EQ(
      c10::backend::DeviceGeneratorImpl::device_type(),
      c10::DeviceType::PrivateUse1);
}

TEST(DeviceGeneratorImpl, TestGetSetOffset) {
  if (!c10::backend::is_available()) {
    GTEST_SKIP() << "NPU is not available";
  }

  auto gen = c10::backend::detail::createGenerator();
  auto npu_gen = at::check_generator<c10::backend::DeviceGeneratorImpl>(gen);
  npu_gen->set_offset(100);
  EXPECT_EQ(npu_gen->get_offset(), 100);
}

TEST(DeviceGeneratorImpl, TestGetSetPhiloxOffset) {
  if (!c10::backend::is_available()) {
    GTEST_SKIP() << "NPU is not available";
  }

  auto gen = c10::backend::detail::createGenerator();
  auto npu_gen = at::check_generator<c10::backend::DeviceGeneratorImpl>(gen);
  npu_gen->set_philox_offset_per_thread(200);
  EXPECT_EQ(npu_gen->philox_offset_per_thread(), 200);
}
