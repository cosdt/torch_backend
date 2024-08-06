#include <gtest/gtest.h>

#include "csrc/core/guard/PrivateUse1GuardImpl.h"

TEST(DeviceGuardTest, StaticType) {
  EXPECT_EQ(
      c10::backend::impl::PrivateUse1GuardImpl::static_type,
      c10::DeviceType::PrivateUse1);
}
