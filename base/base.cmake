# common dependencies for all demo projects

# Dependency: OpenXR
include("${CMAKE_CURRENT_LIST_DIR}/../external/openxr/openxr.cmake")

if (ANDROID AND NOT Vulkan_LIBRARY)
    find_library(Vulkan_LIBRARY NAMES vulkan)
endif()
#message(FATAL_ERROR "READBACK_USE_GPU ${READBACK_USE_GPU}")

if (READBACK_USE_OPENGL)
    if (NOT ANDROID)
        message(FATAL_ERROR "The OpenGL ES readback path now only supports Android builds.")
    endif()

    set(XR_USE_GRAPHICS_API_OPENGL_ES TRUE)
    add_definitions(-DXR_USE_GRAPHICS_API_OPENGL_ES)
    message(STATUS "Enabling Android OpenGL ES support")

    # Only bring in the OpenGL wrapper now located under oxr_utils
    set(COMMON_LIB_SRCS ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/gfxwrapper_opengl.c)
    add_library(common_lib ${COMMON_LIB_SRCS})
    target_include_directories(common_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/oxr_utils)
    target_link_libraries(common_lib PUBLIC EGL GLESv3)
else()
    find_package(Vulkan)
    if(Vulkan_FOUND)
        set(XR_USE_GRAPHICS_API_VULKAN TRUE)
        add_definitions(-DXR_USE_GRAPHICS_API_VULKAN)
        message(STATUS "Enabling Vulkan support")
    elseif(BUILD_ALL_EXTENSIONS)
        message(FATAL_ERROR "Vulkan headers not found")
    endif()
endif()


set(BASE_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/d3d_common.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/graphicsplugin_d3d11.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/graphicsplugin_d3d12.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/graphicsplugin_factory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/graphicsplugin_opengl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/graphicsplugin_opengles.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/graphicsplugin_vulkan.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/graphicsplugin_metal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/logger.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/platformplugin_android.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/platformplugin_factory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/platformplugin_posix.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/platformplugin_win32.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/pch.cpp
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/openxr_program.cpp
)
set(VULKAN_SHADERS
    ${CMAKE_CURRENT_LIST_DIR}/vulkan_shaders/frag.glsl
    ${CMAKE_CURRENT_LIST_DIR}/vulkan_shaders/vert.glsl
)
set(SECUREMR_UTILS_SRCS "")

if (USE_SECURE_MR_UTILS)
    list(APPEND SECUREMR_UTILS_SRCS
        ${CMAKE_CURRENT_LIST_DIR}/securemr_utils/pipeline.cpp
        ${CMAKE_CURRENT_LIST_DIR}/securemr_utils/rendercommand.cpp
        ${CMAKE_CURRENT_LIST_DIR}/securemr_utils/serialization.cpp
        ${CMAKE_CURRENT_LIST_DIR}/securemr_utils/session.cpp
        ${CMAKE_CURRENT_LIST_DIR}/securemr_utils/tensor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/securemr_utils/readback_async.cpp
            ${CMAKE_CURRENT_LIST_DIR}/securemr_utils/readback.cpp
        ${CMAKE_CURRENT_LIST_DIR}/securemr_utils/utils.cpp
    )
endif()
set(BASE_THIRD_PARTY_DIRS ${CMAKE_CURRENT_LIST_DIR}/third_party)

# Dependency: nlohmann_json
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
)
FetchContent_GetProperties(nlohmann_json)
if(NOT nlohmann_json_POPULATED)
    FetchContent_Populate(nlohmann_json)
endif()

add_library(
    ${PROJECT_NAME} MODULE
    ${BASE_SRCS}
    ${SAMPLE_SRCS}
    ${SECUREMR_UTILS_SRCS}
    ${VULKAN_SHADERS}
    ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c
)
target_link_libraries(
        ${PROJECT_NAME} PRIVATE
        android
        log
        OpenXR::openxr_loader
        ${Vulkan_LIBRARY}
)


if (READBACK_USE_OPENGL)
target_link_libraries(
        ${PROJECT_NAME} PRIVATE
        common_lib
)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    "${ANDROID_NDK}/sources/android/native_app_glue"
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/vulkan_shaders
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils
    ${SAMPLE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${Vulkan_INCLUDE_DIRS}
    ${nlohmann_json_SOURCE_DIR}/single_include
    ${BASE_THIRD_PARTY_DIRS}
    ${THIRD_PARTY_DIRS}
)

# Shader compilation for **client**
# No shader needed for SecureMR stuff
include("${CMAKE_CURRENT_LIST_DIR}/../scripts/compile_glsl.cmake")
compile_glsl(run_glsl_compiles ${VULKAN_SHADERS})
if(GLSLANG_VALIDATOR AND NOT GLSLC_COMMAND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_GLSLANGVALIDATOR)
endif()
add_dependencies(${PROJECT_NAME} run_glsl_compiles)
if (NOT READBACK_USE_GPU)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
            DEFAULT_GRAPHICS_PLUGIN_VULKAN
            XR_USE_PLATFORM_ANDROID
            XR_READBACK_USE_CPU
            XR_USE_GRAPHICS_API_VULKAN)

endif ()
if (READBACK_USE_VULKAN)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    DEFAULT_GRAPHICS_PLUGIN_VULKAN
    XR_USE_PLATFORM_ANDROID
    XR_USE_GRAPHICS_API_VULKAN)
endif ()
if (READBACK_USE_OPENGL)
target_compile_definitions(${PROJECT_NAME} PRIVATE
        DEFAULT_GRAPHICS_PLUGIN_OPENGLES
        XR_USE_PLATFORM_ANDROID
        XR_USE_GRAPHICS_API_OPENGL_ES)
endif ()
