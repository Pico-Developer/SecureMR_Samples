cmake_minimum_required(VERSION 3.22)
project(yolo_det)

set(SAMPLE_SRCS
        ${CMAKE_CURRENT_LIST_DIR}/cpp/yolo_object_detection.cpp
)
set(SAMPLE_DIR ${CMAKE_CURRENT_LIST_DIR}/cpp)
set(USE_SECURE_MR_UTILS ON)

include(../../base/base.cmake)
