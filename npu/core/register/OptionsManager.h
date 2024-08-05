#pragma once

#include <map>
#include <string>
#include <unordered_map>

#include "csrc/core/Macros.h"
#include "npu/core/NPUException.h"

namespace c10_npu {
namespace option {

class OptionsManager {
 public:
  static bool IsResumeModeEnable();
  static bool CheckInfNanModeEnable();
  static bool CheckBlockingEnable();
  static bool CheckCombinedOptimizerEnable();
  static bool CheckAclDumpDateEnable();
  static int32_t GetACLExecTimeout();
  C10_BACKEND_API static bool isACLGlobalLogOn(aclLogLevel level);
  static int64_t GetRankId();
  static bool CheckGeInitDisable();

 private:
  static int GetBoolTypeOption(const char* env_str, int defaultVal = 0);
  static std::unordered_map<std::string, std::string> ParsePerfConfig(
      const std::string& config);
};

} // namespace option
} // namespace c10_npu
