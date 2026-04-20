cmake_minimum_required(VERSION 3.22)
project(mnistwild)

set(SAMPLE_SRCS ${CMAKE_CURRENT_LIST_DIR}/cpp/mnistwild.cpp)
set(SAMPLE_DIR ${CMAKE_CURRENT_LIST_DIR}/cpp)
set(USE_SECURE_MR_UTILS ON)
# set(READBACK_USE_OPENGL ON)

include(../../base/base.cmake)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
