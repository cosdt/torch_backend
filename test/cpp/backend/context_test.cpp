#include <gtest/gtest.h>
#include "csrc/backend/Context.h"

TEST(NPUContextTest, TestGetDeviceProperties) {
  if (!c10::backend::is_available()) {
    GTEST_SKIP() << "NPU is not available";
  }

  auto prop = c10::backend::getDeviceProperties(0);
  EXPECT_NE(prop->name, " ");
}
