#include <gtest/gtest.h>
#include "csrc/npu/NPUGeneratorImpl.h"

class NPUGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    at::globalContext().lazyInitPrivateUse1();
  }
};

TEST(NPUGeneratorTest, TestSingletonDefaultGenerator) {
  auto gen = at_npu::detail::getDefaultNPUGenerator();
  auto other = at_npu::detail::getDefaultNPUGenerator();
  EXPECT_EQ(gen, other);
}

TEST(NPUGeneratorTest, TestCloning) {
  auto gen1 = at_npu::detail::createNPUGenerator();
  auto npu_gen1 = at::check_generator<at_npu::NPUGeneratorImpl>(gen1);

  auto gen2 = at_npu::detail::createNPUGenerator();
  gen2 = gen1.clone();
  auto npu_gen2 = at::check_generator<at_npu::NPUGeneratorImpl>(gen2);

  EXPECT_EQ(npu_gen1->current_seed(), npu_gen2->current_seed());
  EXPECT_EQ(
      npu_gen1->philox_offset_per_thread(),
      npu_gen2->philox_offset_per_thread());
}

TEST(NPUGeneratorTest, TestGetSetCurrentSeed) {
  auto gen = at_npu::detail::createNPUGenerator();
  auto npu_gen = at::check_generator<at_npu::NPUGeneratorImpl>(gen);
  npu_gen->set_current_seed(10);
  EXPECT_EQ(npu_gen->current_seed(), 10);
}

TEST(NPUGeneratorTest, TestDeviceType) {
  EXPECT_EQ(
      at_npu::NPUGeneratorImpl::device_type(), c10::DeviceType::PrivateUse1);
}

TEST(NPUGeneratorTest, TestGetSetOffset) {
  auto gen = at_npu::detail::createNPUGenerator();
  auto npu_gen = at::check_generator<at_npu::NPUGeneratorImpl>(gen);
  npu_gen->set_offset(100);
  EXPECT_EQ(npu_gen->get_offset(), 100);
}

TEST(NPUGeneratorTest, TestGetSetPhiloxOffset) {
  auto gen = at_npu::detail::createNPUGenerator();
  auto npu_gen = at::check_generator<at_npu::NPUGeneratorImpl>(gen);
  npu_gen->set_philox_offset_per_thread(200);
  EXPECT_EQ(npu_gen->philox_offset_per_thread(), 200);
}