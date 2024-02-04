file(GLOB igo_sources "*.c")

message(STATUS "igo sources = ${igo_sources}")

add_library(igo SHARED ${igo_sources})

target_link_libraries(igo "/opt/intel/oneapi/vtune/latest/sdk/lib64/libadvisor.a")
target_link_libraries(igo "/opt/intel/oneapi/vtune/latest/sdk/lib64/libittnotify.a")
target_link_libraries(igo "/opt/intel/oneapi/vtune/latest/sdk/lib64/libjitprofiling.a")


