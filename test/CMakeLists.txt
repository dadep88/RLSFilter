cmake_minimum_required(VERSION 3.19)
#project(test_rls_filter LANGUAGES CXX)

#Setup testing
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})

#Add test cpp file
add_executable("${PROJECT_NAME}_test" test_rls_filter.cpp)
target_include_directories("${PROJECT_NAME}_test" PRIVATE include)

#Link test executable against gtest &gtest_main
target_link_libraries("${PROJECT_NAME}_test" ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY})
target_link_libraries("${PROJECT_NAME}_test" ${CMAKE_PROJECT_NAME})
target_link_libraries( "${PROJECT_NAME}_test" pthread)

add_test(NAME ${PROJECT_NAME} COMMAND "${PROJECT_NAME}_test")