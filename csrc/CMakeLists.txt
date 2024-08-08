# Generated Python Function
FILE(GLOB GENERATED_SRCS
    aten/generated/python_functions_*)
LIST(APPEND PYTHON_SRCS ${GENERATED_SRCS})

# Pass to parent
set(PYTHON_SRCS ${PYTHON_SRCS} PARENT_SCOPE)


# CSRC
file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

FILE(GLOB EXCLUDE
    aten/generated/python_functions*
    aten/generated/VariableTypeEverything.cpp
    aten/generated/ADInplaceOrViewTypeEverything.cpp
    aten/generated/RegisterFunctionalizationEverything.cpp)

FOREACH(ITEM ${EXCLUDE})
    LIST(REMOVE_ITEM SOURCE_FILES ${ITEM})
ENDFOREACH()

LIST(APPEND BACKEND_SRCS ${SOURCE_FILES})

# Pass to parent
set(BACKEND_SRCS ${BACKEND_SRCS} PARENT_SCOPE)
