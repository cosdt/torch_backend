#include <gtest/gtest.h>
#include "csrc/npu/NPUContext.h"

TEST(NPUContextTest, TestGetDeviceProperties) {
  if (!c10::npu::is_available()) {
    GTEST_SKIP() << "NPU is not available";
  }

  auto prop = c10::npu::getDeviceProperties(0);
  EXPECT_NE(prop->name, " ");
}
