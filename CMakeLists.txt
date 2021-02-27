cmake_minimum_required(VERSION 3.19)
project(RLSFilter LANGUAGES CXX VERSION 0.0.0 DESCRIPTION "C++ RLS filter")

set(CMAKE_CXX_STANDARD 20)

include(cmake/SourcesAndHeaders.cmake)

add_library(${PROJECT_NAME} ${sources} ${headers})

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER include)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PRIVATE src)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

target_link_libraries(
        ${PROJECT_NAME}
        PUBLIC
        Eigen3::Eigen
)

if(NOT ${PROJECT_NAME}_CLANG_FORMAT_BINARY)
    find_program(${PROJECT_NAME}_CLANG_FORMAT_BINARY clang-format)
endif()


add_custom_target(clang-format
        COMMAND ${${PROJECT_NAME}_CLANG_FORMAT_BINARY}
        -i ${ALL_SOURCE_FILES} ${sources} ${exe_sources} ${headers})

add_subdirectory(test)











