#include "npu/core/NPUException.h"
#include "csrc/npu/NPUFunctions.h"
#include "npu/core/register/OptionsManager.h"

std::unordered_map<SubModule, std::string> submoduleMap = {
    {SubModule::PTA, "PTA"},
    {SubModule::OPS, "OPS"},
    {SubModule::DIST, "DIST"},
    {SubModule::GRAPH, "GRAPH"},
    {SubModule::PROF, "PROF"}};

std::unordered_map<ErrCode, std::string> errCodeMap = {
    {ErrCode::SUC, "success"},
    {ErrCode::PARAM, "invalid parameter"},
    {ErrCode::TYPE, "invalid type"},
    {ErrCode::VALUE, "invalid value"},
    {ErrCode::PTR, "invalid pointer"},
    {ErrCode::INTERNAL, "internal error"},
    {ErrCode::MEMORY, "memory error"},
    {ErrCode::NOT_SUPPORT, "feature not supported"},
    {ErrCode::NOT_FOUND, "resource not found"},
    {ErrCode::UNAVAIL, "resource unavailable"},
    {ErrCode::SYSCALL, "system call failed"},
    {ErrCode::TIMEOUT, "timeout error"},
    {ErrCode::PERMISSION, "permission error"},
    {ErrCode::ACL, "call acl api failed"},
    {ErrCode::HCCL, "call hccl api failed"},
    {ErrCode::GE, "call ge api failed"}};

std::string formatErrorCode(SubModule submodule, ErrCode errorCode) {
  std::ostringstream oss;
  c10::DeviceIndex deviceIndex = -1;
  c10_npu::GetDevice(&deviceIndex);
  auto rank_id = c10_npu::option::OptionsManager::GetRankId();
  oss << "\n[ERROR] " << getCurrentTimestamp() << " (PID:" << getpid()
      << ", Device:" << deviceIndex << ", RankID:" << rank_id << ") ";
  oss << "ERR" << std::setw(2) << std::setfill('0')
      << static_cast<int>(submodule);
  oss << std::setw(3) << std::setfill('0') << static_cast<int>(errorCode);
  oss << " " << submoduleMap[submodule] << " " << errCodeMap[errorCode];

  return oss.str();
}

static std::string getCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
      now.time_since_epoch());

  std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
  std::tm* timeInfo = std::localtime(&currentTime);

  auto milli_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(micros).count() %
      1000;
  auto micro_time = micros.count() % 1000;

  std::ostringstream oss;
  oss << std::put_time(timeInfo, "%Y-%m-%d-%H:%M:%S");
  return oss.str();
}

namespace c10_npu {

const char* c10_npu_get_error_message() {
  return c10_npu::acl::AclGetErrMsg();
}

} // namespace c10_npu
