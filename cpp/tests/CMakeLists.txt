add_executable(test_brimfile test_brimfile.cpp)

target_link_libraries(test_brimfile PRIVATE brimfile_cpp)
target_include_directories(test_brimfile PRIVATE ${Python3_INCLUDE_DIRS})

add_test(NAME brimfile_cpp_tests COMMAND test_brimfile)

# Make sure tests can find the Python interpreter and the brimfile package
# that is installed in the current environment.
set_tests_properties(brimfile_cpp_tests PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/../src"
)
