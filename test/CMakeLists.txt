cmake_minimum_required(VERSION 3.19)
project(test_rls_filter LANGUAGES CXX)

# Setup testing
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})

# Add test cpp file
add_executable(${PROJECT_NAME}
        test_rls_filter.cpp
        )

# Link test executable against gtest & gtest_main
target_link_libraries(${PROJECT_NAME} ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY})

add_test(
        NAME ${PROJECT_NAME}
        COMMAND ${PROJECT_NAME}
)