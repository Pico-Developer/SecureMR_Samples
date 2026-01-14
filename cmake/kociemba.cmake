# Fetch and build Kociemba C solver via CMake FetchContent

include(FetchContent)

if(NOT DEFINED KOCIEEMBA_GIT_REPOSITORY)
    set(KOCIEEMBA_GIT_REPOSITORY https://github.com/muodov/kociemba.git)
endif()

# You can override this at configure time: -DKOCIEMBA_GIT_TAG=<tag-or-commit>
if(NOT DEFINED KOCIEMBA_GIT_TAG)
    # A known stable commit; update as needed
    set(KOCIEMBA_GIT_TAG master)
endif()

set(FETCHCONTENT_QUIET OFF)
FetchContent_Declare(
    kociemba_src
    GIT_REPOSITORY ${KOCIEEMBA_GIT_REPOSITORY}
    GIT_TAG        ${KOCIEMBA_GIT_TAG}
)

FetchContent_GetProperties(kociemba_src)
if(NOT kociemba_src_POPULATED)
    message(STATUS "Fetching Kociemba from ${KOCIEEMBA_GIT_REPOSITORY} @ ${KOCIEMBA_GIT_TAG}")
    FetchContent_Populate(kociemba_src)
endif()

set(_KOCIEMBA_C_DIR ${kociemba_src_SOURCE_DIR}/kociemba/ckociemba)
if(NOT EXISTS ${_KOCIEMBA_C_DIR}/search.c)
    message(FATAL_ERROR "Kociemba ckociemba sources not found under ${_KOCIEMBA_C_DIR}")
endif()

add_library(kociemba_c STATIC
    ${_KOCIEMBA_C_DIR}/coordcube.c
    ${_KOCIEMBA_C_DIR}/cubiecube.c
    ${_KOCIEMBA_C_DIR}/facecube.c
    ${_KOCIEMBA_C_DIR}/search.c
    ${_KOCIEMBA_C_DIR}/prunetable_helpers.c
)

target_include_directories(kociemba_c PUBLIC ${_KOCIEMBA_C_DIR}/include)

