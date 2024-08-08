#include <gtest/gtest.h>
#include "csrc/backend/NPUContext.h"
#include "csrc/backend/NPUGeneratorImpl.h"

TEST(NPUGeneratorImpl, TestSingletonDefaultGenerator) {
  if (!c10::backend::is_available()) {
    GTEST_SKIP() << "NPU is not available";
  }

  auto gen = c10::backend::detail::getDefaultNPUGenerator();
  auto other = c10::backend::detail::getDefaultNPUGenerator();
  EXPECT_EQ(gen, other);
}

TEST(NPUGeneratorImpl, TestCloning) {
  if (!c10::backend::is_available()) {
    GTEST_SKIP() << "NPU is not available";
  }

  auto gen1 = c10::backend::detail::createNPUGenerator();
  auto npu_gen1 = at::check_generator<c10::backend::NPUGeneratorImpl>(gen1);

  auto gen2 = c10::backend::detail::createNPUGenerator();
  gen2 = gen1.clone();
  auto npu_gen2 = at::check_generator<c10::backend::NPUGeneratorImpl>(gen2);

  EXPECT_EQ(npu_gen1->current_seed(), npu_gen2->current_seed());
  EXPECT_EQ(
      npu_gen1->philox_offset_per_thread(),
      npu_gen2->philox_offset_per_thread());
}

TEST(NPUGeneratorImpl, TestGetSetCurrentSeed) {
  if (!c10::backend::is_available()) {
    GTEST_SKIP() << "NPU is not available";
  }

  auto gen = c10::backend::detail::createNPUGenerator();
  auto npu_gen = at::check_generator<c10::backend::NPUGeneratorImpl>(gen);
  npu_gen->set_current_seed(10);
  EXPECT_EQ(npu_gen->current_seed(), 10);
}

TEST(NPUGeneratorImpl, TestDeviceType) {
  EXPECT_EQ(
      c10::backend::NPUGeneratorImpl::device_type(),
      c10::DeviceType::PrivateUse1);
}

TEST(NPUGeneratorImpl, TestGetSetOffset) {
  if (!c10::backend::is_available()) {
    GTEST_SKIP() << "NPU is not available";
  }

  auto gen = c10::backend::detail::createNPUGenerator();
  auto npu_gen = at::check_generator<c10::backend::NPUGeneratorImpl>(gen);
  npu_gen->set_offset(100);
  EXPECT_EQ(npu_gen->get_offset(), 100);
}

TEST(NPUGeneratorImpl, TestGetSetPhiloxOffset) {
  if (!c10::backend::is_available()) {
    GTEST_SKIP() << "NPU is not available";
  }

  auto gen = c10::backend::detail::createNPUGenerator();
  auto npu_gen = at::check_generator<c10::backend::NPUGeneratorImpl>(gen);
  npu_gen->set_philox_offset_per_thread(200);
  EXPECT_EQ(npu_gen->philox_offset_per_thread(), 200);
}
