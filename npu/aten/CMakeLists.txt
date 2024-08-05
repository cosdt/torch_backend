FILE(GLOB _ATEN_SRCS
    common/*.cpp
    mirror/*.cpp
    ops/*.cpp
    ops/base/aclops/*.cpp
    ops/base/opapi/*.cpp
    ops/latest/aclops/*.cpp
    ops/latest/op_api/*.cpp
    ops/latest/op_api/sparse/*.cpp
    utils/*.cpp
    utils/custom_functions/aclops/*.cpp
    utils/custom_functions/opapi/*.cpp
    *.cpp
    )

LIST(APPEND ATEN_SRCS ${_ATEN_SRCS})
# Pass to parent
set(ATEN_SRCS ${ATEN_SRCS} PARENT_SCOPE)
