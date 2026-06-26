cmake_minimum_required(VERSION 3.16)
project(navcore_eskf VERSION 2.0.0 LANGUAGES CXX)

# ── C++17 Standard ───────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build type defaults ──────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")

# ── Eigen3 ───────────────────────────────────────────────────
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")

# ── Library ──────────────────────────────────────────────────
add_library(navcore_eskf STATIC
    src/eskf_core.cpp
)

target_include_directories(navcore_eskf
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(navcore_eskf
    PUBLIC
        Eigen3::Eigen
        $<$<PLATFORM_ID:Linux>:pthread>
)

# Enable RT scheduling on Linux/ARM (SCHED_FIFO for hard real-time)
target_compile_definitions(navcore_eskf PUBLIC
    $<$<PLATFORM_ID:Linux>:ENABLE_RT_SCHEDULING>
)

# ── Test executable (optional) ──────────────────────────────
option(BUILD_TESTS "Build test executable" ON)

if(BUILD_TESTS)
    add_executable(test_eskf
        tests/test_eskf.cpp
    )
    target_link_libraries(test_eskf PRIVATE navcore_eskf)
endif()

# ── Install ──────────────────────────────────────────────────
install(TARGETS navcore_eskf
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)

# ── Platform info ────────────────────────────────────────────
message(STATUS "")
message(STATUS "NavCore ESKF C++ Port v${PROJECT_VERSION}")
message(STATUS "  C++ Standard : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Architecture : ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Features     : GPS, ZUPT, AdaptiveQ, BaroDrift, MagAutoCal")
message(STATUS "  RT scheduling: $<$<PLATFORM_ID:Linux>:ENABLED>")
message(STATUS "")
