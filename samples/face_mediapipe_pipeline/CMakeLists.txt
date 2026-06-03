cmake_minimum_required(VERSION 3.22)
project(face_mediapipe_pipeline)

set(SAMPLE_SRCS
        ${CMAKE_CURRENT_LIST_DIR}/cpp/face_mediapipe_pipeline.cpp
)
set(SAMPLE_DIR ${CMAKE_CURRENT_LIST_DIR}/cpp)
set(USE_SECURE_MR_UTILS ON)
set(THIRD_PARTY_DIRS "")

include(../../base/base.cmake)
