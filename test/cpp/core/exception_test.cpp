#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "csrc/core/Exception.h"

namespace {

template <class Functor>
inline void expectThrows(Functor&& functor, const char* expectedMessage) {
  try {
    std::forward<Functor>(functor)();
  } catch (const c10::Error& e) {
    EXPECT_THAT(
        e.what_without_backtrace(), ::testing::HasSubstr(expectedMessage));
    return;
  }
  ADD_FAILURE() << "Expected to throw exception with message \""
                << expectedMessage << "\" but didn't throw";
}
} // namespace

TEST(ExceptionTest, TestPrintWarning) {
  TORCH_BACKEND_WARN("Backend warning.");
}

TEST(ExceptionTest, TestFormatWarning) {
  EXPECT_NO_THROW(TORCH_BACKEND_FORMAT_WARN(1, "Test format warning."));
}

TEST(ExceptionTest, TestFormatError) {
  expectThrows(
      []() { TORCH_BACKEND_FORMAT_ERROR(1, "Test format error."); },
      "\nBackend error, error code is 1\nTest format error.");
}