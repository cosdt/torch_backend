#pragma once
#include <c10/util/Exception.h>

void warn_(const ::c10::Warning& warning);

#define TORCH_BACKEND_WARN(...)                              \
  warn_(::c10::Warning(                                      \
      ::c10::UserWarning(),                                  \
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
      ::c10::str(__VA_ARGS__),                               \
      false));

#define TORCH_BACKEND_WARN_ONCE(...)                                          \
  C10_UNUSED static const auto C10_ANONYMOUS_VARIABLE(TORCH_NPU_WARN_ONCE_) = \
      [&] {                                                                   \
        TORCH_BACKEND_WARN(__VA_ARGS__);                                      \
        return true;                                                          \
      }()

#define TORCH_BACKEND_FORMAT_ERROR(error_code, error_code_map, ...) \
  TORCH_CHECK(                                                      \
      false,                                                        \
      __func__,                                                     \
      ":",                                                          \
      __FILE__,                                                     \
      ":",                                                          \
      __LINE__,                                                     \
      "\nBackend Error, error code is ",                            \
      error_code,                                                   \
      (error_code_map.find(error_code) != error_code_map.end()      \
           ? "\n[Error]: " + error_code_map[error_code]             \
           : "."),                                                  \
      "\n",                                                         \
      ::c10::str(__VA_ARGS__));

#define TORCH_BACKEND_FORMAT_WARN(error_code, error_code_map, ...) \
  TORCH_BACKEND_WARN(                                              \
      "\nBackend warning, error code is ",                         \
      error_code,                                                  \
      "[Error]: ",                                                 \
      (error_code_map.find(error_code) != error_code_map.end()     \
           ? "\n[Error]: " + error_code_map[error_code]            \
           : "."),                                                 \
      "\n",                                                        \
      ::c10::str(__VA_ARGS__));
