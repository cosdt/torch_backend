#include <string>

#include "npu/core/NPUException.h"
#include "npu/core/register/OptionRegister.h"
#include "npu/core/register/OptionsManager.h"

namespace c10_npu {
namespace option {

using namespace std;

bool OptionsManager::IsResumeModeEnable() {
  const static bool isResumeModeEnable = []() -> bool {
    int32_t enable = OptionsManager::GetBoolTypeOption("RESUME_MODE_ENABLE", 0);
    return enable != 0;
  }();
  return isResumeModeEnable;
}

bool OptionsManager::CheckInfNanModeEnable() {
  const static bool checkInfNanModeEnable = []() -> bool {
    int32_t enable =
        OptionsManager::GetBoolTypeOption("INF_NAN_MODE_ENABLE", 1);
    return enable != 0;
  }();
  return checkInfNanModeEnable;
}

bool OptionsManager::CheckBlockingEnable() {
  const static bool checkBlockingEnable = []() -> bool {
    int32_t blocking_enable =
        OptionsManager::GetBoolTypeOption("ASCEND_LAUNCH_BLOCKING", 0);
    return blocking_enable != 0;
  }();
  return checkBlockingEnable;
}

bool OptionsManager::CheckCombinedOptimizerEnable() {
  const static bool checkCombinedOptimizerEnable = []() -> bool {
    int32_t combined_optimize =
        OptionsManager::GetBoolTypeOption("COMBINED_ENABLE");
    return combined_optimize != 0;
  }();
  return checkCombinedOptimizerEnable;
}

bool OptionsManager::CheckAclDumpDateEnable() {
  const static bool checkAclDumpDateEnable = []() -> bool {
    int32_t acl_dump_data = OptionsManager::GetBoolTypeOption("ACL_DUMP_DATA");
    return acl_dump_data != 0;
  }();
  if (checkAclDumpDateEnable) {
    TORCH_NPU_WARN_ONCE(
        "The environment variable ACL_DUMP_DATA has been deprecated, "
        "please use torch_npu.npu.init_dump() instead");
  }
  return checkAclDumpDateEnable;
}

int OptionsManager::GetBoolTypeOption(const char* env_str, int defaultVal) {
  char* env_val = std::getenv(env_str);
  int64_t envFlag =
      (env_val != nullptr) ? strtol(env_val, nullptr, 10) : defaultVal;
  return (envFlag != 0) ? 1 : 0;
}

int32_t OptionsManager::GetACLExecTimeout() {
  char* env_val = std::getenv("ACL_STREAM_TIMEOUT");
  int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : -1;
  return static_cast<int32_t>(envFlag);
}

bool OptionsManager::isACLGlobalLogOn(aclLogLevel level) {
  const static int getACLGlobalLogLevel = []() -> int {
    char* env_val = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");
    int64_t envFlag =
        (env_val != nullptr) ? strtol(env_val, nullptr, 10) : ACL_ERROR;
    return static_cast<int>(envFlag);
  }();
  return (getACLGlobalLogLevel <= level);
}

int64_t OptionsManager::GetRankId() {
  char* rankId_val = std::getenv("RANK");
  int64_t rankId =
      (rankId_val != nullptr) ? strtol(rankId_val, nullptr, 10) : -1;
  return rankId;
}

bool OptionsManager::CheckGeInitDisable() {
  const static bool Check_Ge_Init_Disable = []() -> bool {
    int32_t ge_init_disable =
        OptionsManager::GetBoolTypeOption("GE_INIT_DISABLE");
    return ge_init_disable != 0;
  }();
  if (Check_Ge_Init_Disable) {
    TORCH_NPU_WARN_ONCE(
        "The environment variable GE_INIT_DISABLE has been enabled, "
        "this switch is only used for single operator simulation");
  }
  return Check_Ge_Init_Disable;
}

std::unordered_map<std::string, std::string> OptionsManager::ParsePerfConfig(
    const std::string& config) {
  std::unordered_map<std::string, std::string> config_map;
  size_t start = 0;
  size_t end = config.find(',');

  while (end != std::string::npos) {
    std::string item = config.substr(start, end - start);
    size_t delimiter_pos = item.find(':');
    if (delimiter_pos != std::string::npos) {
      std::string key = item.substr(0, delimiter_pos);
      std::string value = item.substr(delimiter_pos + 1);
      config_map[key] = value;
    }
    start = end + 1;
    end = config.find(',', start);
  }

  // Handle the last item
  std::string last_item = config.substr(start);
  size_t delimiter_pos = last_item.find(':');
  if (delimiter_pos != std::string::npos) {
    std::string key = last_item.substr(0, delimiter_pos);
    std::string value = last_item.substr(delimiter_pos + 1);
    config_map[key] = value;
  }

  return config_map;
}

} // namespace option
} // namespace c10_npu
