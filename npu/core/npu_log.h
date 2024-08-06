#pragma once

#include <stdio.h>
#include <iostream>
#include <string>
#include "npu/acl/include/acl/acl_base.h"
#include "npu/core/register/OptionsManager.h"

#define NPUStatus std::string
#define SUCCESS "SUCCESS"
#define INTERNEL_ERROR "INTERNEL_ERROR"
#define PARAM_ERROR "PARAM_ERROR"
#define ALLOC_ERROR "ALLOC_ERROR"
#define FAILED "FAILED"

#define ASCEND_LOGE(fmt, ...)                                            \
  do {                                                                   \
    if (c10::npu::option::OptionsManager::isACLGlobalLogOn(ACL_ERROR)) { \
      aclAppLog(                                                         \
          ACL_ERROR,                                                     \
          __FILENAME__,                                                  \
          __FUNCTION__,                                                  \
          __LINE__,                                                      \
          "[PTA]:" #fmt,                                                 \
          ##__VA_ARGS__);                                                \
    }                                                                    \
  } while (0);
#define ASCEND_LOGW(fmt, ...)                                              \
  do {                                                                     \
    if (c10::npu::option::OptionsManager::isACLGlobalLogOn(ACL_WARNING)) { \
      aclAppLog(                                                           \
          ACL_WARNING,                                                     \
          __FILENAME__,                                                    \
          __FUNCTION__,                                                    \
          __LINE__,                                                        \
          "[PTA]:" #fmt,                                                   \
          ##__VA_ARGS__);                                                  \
    }                                                                      \
  } while (0);
#define ASCEND_LOGI(fmt, ...)                                           \
  do {                                                                  \
    if (c10::npu::option::OptionsManager::isACLGlobalLogOn(ACL_INFO)) { \
      aclAppLog(                                                        \
          ACL_INFO,                                                     \
          __FILENAME__,                                                 \
          __FUNCTION__,                                                 \
          __LINE__,                                                     \
          "[PTA]:" #fmt,                                                \
          ##__VA_ARGS__);                                               \
    }                                                                   \
  } while (0);
#define ASCEND_LOGD(fmt, ...)                                            \
  do {                                                                   \
    if (c10::npu::option::OptionsManager::isACLGlobalLogOn(ACL_DEBUG)) { \
      aclAppLog(                                                         \
          ACL_DEBUG,                                                     \
          __FILENAME__,                                                  \
          __FUNCTION__,                                                  \
          __LINE__,                                                      \
          "[PTA]:" #fmt,                                                 \
          ##__VA_ARGS__);                                                \
    }                                                                    \
  } while (0);
