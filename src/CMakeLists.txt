file(GLOB igo_sources "*.cpp")

message(STATUS "igo sources = ${igo_sources}")

add_library(igo SHARED ${igo_sources})


