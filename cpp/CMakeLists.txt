cmake_minimum_required(VERSION 3.15)
project(brimfile_cpp VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Locate the Python interpreter and development headers.
# ---------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

message(STATUS "Python3 version:   ${Python3_VERSION}")
message(STATUS "Python3 includes:  ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")

# ---------------------------------------------------------------------------
# brimfile_cpp library
# ---------------------------------------------------------------------------
add_library(brimfile_cpp STATIC
    src/brimfile.cpp
)

target_include_directories(brimfile_cpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${Python3_INCLUDE_DIRS}
)

target_link_libraries(brimfile_cpp
    PUBLIC
        ${Python3_LIBRARIES}
)

# ---------------------------------------------------------------------------
# Optional: tests
# ---------------------------------------------------------------------------
option(BRIMFILE_CPP_BUILD_TESTS "Build the brimfile C++ wrapper tests" ON)
if(BRIMFILE_CPP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Optional: examples
# ---------------------------------------------------------------------------
option(BRIMFILE_CPP_BUILD_EXAMPLES "Build the brimfile C++ wrapper examples" ON)
if(BRIMFILE_CPP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ---------------------------------------------------------------------------
# Install rules (optional, for packaging)
# ---------------------------------------------------------------------------
install(TARGETS brimfile_cpp
    EXPORT brimfile_cpp_targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
