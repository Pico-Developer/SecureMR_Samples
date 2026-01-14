cmake_minimum_required(VERSION 3.22)
project(readback)

set(SAMPLE_SRCS
        ${CMAKE_CURRENT_LIST_DIR}/cpp/readback_file.cpp
        ${CMAKE_CURRENT_LIST_DIR}/cpp/readback_vulkan.cpp
        ${CMAKE_CURRENT_LIST_DIR}/cpp/readback_opengl.cpp
)
set(SAMPLE_DIR ${CMAKE_CURRENT_LIST_DIR}/cpp)
set(USE_SECURE_MR_UTILS ON)
set(THIRD_PARTY_DIRS ${CMAKE_CURRENT_LIST_DIR}/third_party)

include(../../base/base.cmake)
