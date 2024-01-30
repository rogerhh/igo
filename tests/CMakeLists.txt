# file(GLOB test_sources "*.c")
# foreach(test_file ${test_sources})
#     get_filename_component(test_exe ${test_file} NAME_WLE)
#     add_executable(${test_exe} ${test_file})
#     target_link_libraries(${test_exe} PRIVATE igo)
# endforeach()

enable_testing()

include(GoogleTest)

add_library(test_utils utils.c)

file(GLOB gtest_sources "test*.cpp")
foreach(gtest_file ${gtest_sources})
    get_filename_component(gtest_exe ${gtest_file} NAME_WLE)
    add_executable(${gtest_exe} ${gtest_file})
    target_link_libraries(${gtest_exe} PRIVATE GTest::gtest_main igo test_utils)
    gtest_discover_tests(${gtest_exe})
endforeach()

# add_executable(
#   test_gtest
#   test_gtest.cpp
# )
# target_link_libraries(
#   test_gtest
#   GTest::gtest_main
# )


