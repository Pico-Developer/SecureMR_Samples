# Lightweight OpenCV Android SDK integration for CMake

if(NOT ANDROID)
    message(FATAL_ERROR "OpenCV Android SDK integration only supports Android builds.")
endif()

if(NOT DEFINED OpencvVersion)
    set(OpencvVersion 4.10.0)
endif()

include(FetchContent)
set(_opencv_url https://github.com/opencv/opencv/releases/download/${OpencvVersion}/opencv-${OpencvVersion}-android-sdk.zip)

# Download once into the build tree
set(_opencv_root ${CMAKE_BINARY_DIR}/_deps/opencv)
set(_opencv_zip ${_opencv_root}/opencv-${OpencvVersion}-android-sdk.zip)
set(_opencv_sdk_dir "")

file(MAKE_DIRECTORY ${_opencv_root})

if(NOT EXISTS ${_opencv_root}/OpenCV-android-sdk AND NOT EXISTS ${_opencv_root}/opencv-${OpencvVersion}-android-sdk)
    message(STATUS "Downloading OpenCV Android SDK ${OpencvVersion} ...")
    file(DOWNLOAD ${_opencv_url} ${_opencv_zip} SHOW_PROGRESS STATUS _dl_status TLS_VERIFY ON)
    list(GET _dl_status 0 _dl_status_code)
    if(NOT _dl_status_code EQUAL 0)
        message(FATAL_ERROR "Failed to download OpenCV SDK: ${_dl_status}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xf ${_opencv_zip} WORKING_DIRECTORY ${_opencv_root}
                    RESULT_VARIABLE _unzip_rv)
    if(NOT _unzip_rv EQUAL 0)
        message(FATAL_ERROR "Failed to extract OpenCV SDK archive")
    endif()
endif()

# Detect extracted SDK directory name
foreach(_cand OpenCV-android-sdk opencv-${OpencvVersion}-android-sdk)
    if(EXISTS ${_opencv_root}/${_cand}/sdk/native)
        set(_opencv_sdk_dir ${_opencv_root}/${_cand})
    endif()
endforeach()

if(NOT _opencv_sdk_dir)
    message(FATAL_ERROR "OpenCV SDK directory not found under ${_opencv_root}")
endif()

set(OpenCV_DIR ${_opencv_sdk_dir}/sdk/native)
message(STATUS "OpenCV_DIR=${OpenCV_DIR}")

include_directories(${OpenCV_DIR}/jni/include)

add_library(lib_opencv SHARED IMPORTED GLOBAL)
set_target_properties(lib_opencv PROPERTIES
    IMPORTED_LOCATION ${OpenCV_DIR}/libs/arm64-v8a/libopencv_java4.so)
