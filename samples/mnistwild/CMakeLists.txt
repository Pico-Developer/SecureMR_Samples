cmake_minimum_required(VERSION 3.22)
project(mnistwild)

set(SAMPLE_SRCS ${CMAKE_CURRENT_LIST_DIR}/cpp/mnistwild.cpp)
set(SAMPLE_DIR ${CMAKE_CURRENT_LIST_DIR}/cpp)
set(USE_SECURE_MR_UTILS ON)

include(../../base/base.cmake)

include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
)
FetchContent_GetProperties(nlohmann_json)
if(NOT nlohmann_json_POPULATED)
    FetchContent_Populate(nlohmann_json)
endif()
target_include_directories(${PROJECT_NAME} PRIVATE "${nlohmann_json_SOURCE_DIR}/single_include")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
