cmake_minimum_required(VERSION 3.14)
project(IGO CXX C)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

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
