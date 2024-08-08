# TODO(FFFrog):
# It will be removed when the decoupling is complete
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/acl/libs)

file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/adapter/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/aten/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/framework/*.cpp"
)
LIST(APPEND BACKEND_SRCS ${SOURCE_FILES})

# Pass to parent
set(BACKEND_SRCS ${BACKEND_SRCS} PARENT_SCOPE)
