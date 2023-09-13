project(IGO CXX C)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
add_compile_options(-Werror=int-conversion -Werror=implicit-function-declaration -Werror=incompatible-pointer-types)

find_package(SuiteSparse_config)
find_package(CHOLMOD)

include_directories(${SUITESPARSE_CONFIG_INCLUDE_DIR})
include_directories(${CHOLMOD_INCLUDE_DIR})
include_directories(src)
link_libraries(${SUITESPARSE_CONFIG_LIBRARY})
link_libraries(${CHOLMOD_LIBRARY})

message(STATUS "${CHOLMOD_LIBRARY}")

add_subdirectory(src)
add_subdirectory(tests)
