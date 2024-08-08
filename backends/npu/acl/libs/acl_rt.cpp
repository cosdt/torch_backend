#include "acl/acl_rt.h"

ACL_FUNC_VISIBILITY aclError aclrtGetCurrentContext(aclrtContext* context) {
  return 0;
}

ACL_FUNC_VISIBILITY aclError aclrtSetCurrentContext(aclrtContext context) {
  return 0;
}

ACL_FUNC_VISIBILITY aclError
aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags) {
  return 0;
}

ACL_FUNC_VISIBILITY aclError
aclrtMemset(void* devPtr, size_t maxCount, int32_t value, size_t count) {
  return 0;
}

ACL_FUNC_VISIBILITY aclError aclrtResetOverflowStatus(aclrtStream stream) {
  return 0;
}

ACL_FUNC_VISIBILITY aclError aclrtGetOverflowStatus(
    void* outputAddr,
    size_t outputSize,
    aclrtStream stream) {
  return 0;
}

ACL_FUNC_VISIBILITY const char *aclrtGetSocName() {
  return "";
}

ACL_FUNC_VISIBILITY aclError aclGetCannAttribute(aclCannAttr cannAttr, int32_t *value) {
  return 0;
}

ACL_FUNC_VISIBILITY aclError aclrtMapMem(
    void* virPtr,
    size_t size,
    size_t offset,
    aclrtDrvMemHandle handle,
    uint64_t flags) {
  return 0;
}

ACL_FUNC_VISIBILITY aclError aclrtSetStreamOverflowSwitch(aclrtStream stream, uint32_t flag) {
  return 0;
}

ACL_FUNC_VISIBILITY aclError aclrtReserveMemAddress(
    void** virPtr,
    size_t size,
    size_t alignment,
    void* expectPtr,
    uint64_t flags) {
  return 0;
}

ACL_FUNC_VISIBILITY aclError aclrtReleaseMemAddress(void *virPtr) {
  return 0;
}

ACL_FUNC_VISIBILITY aclError aclrtSetOpExecuteTimeOut(uint32_t timeout) {
  return 0;
}

ACL_FUNC_VISIBILITY aclError aclrtFreePhysical(aclrtDrvMemHandle handle) {
  return 0;
}

ACL_FUNC_VISIBILITY aclError aclrtMallocPhysical(
    aclrtDrvMemHandle* handle,
    size_t size,
    const aclrtPhysicalMemProp* prop,
    uint64_t flags) {
  return 0;
}

ACL_FUNC_VISIBILITY aclError aclrtUnmapMem(void *virPtr) {
  return 0;
}
