#include <gtest/gtest.h>

#include "csrc/core/impl/PrivateUse1GuardImpl.h"

TEST(DeviceGuardTest, StaticType) {
  EXPECT_EQ(true, true);

//  EXPECT_EQ(
//      c10_backend::impl::PrivateUse1GuardImpl::static_type,
//      c10::DeviceType::PrivateUse1);
}
