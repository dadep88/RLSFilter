cmake_minimum_required(VERSION 3.19)
project(RLSFilter LANGUAGES CXX VERSION 0.0.0 DESCRIPTION "C++ RLS filter")
set(CMAKE_CXX_STANDARD 20)

set(RLSFilter_ENABLE_DOXYGEN True)

if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CODE_COVERAGE_VERBOSE True)
    include(cmake/CodeCoverage.cmake)
    APPEND_COVERAGE_COMPILER_FLAGS()
ENDIF()

include(cmake/SourcesAndHeaders.cmake)

add_library(${PROJECT_NAME} INTERFACE)

target_sources(${PROJECT_NAME} INTERFACE ${headers})

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER include)

target_include_directories(${PROJECT_NAME} INTERFACE include)
#target_include_directories(${PROJECT_NAME} PRIVATE src)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

target_link_libraries(
        ${PROJECT_NAME}
        INTERFACE
        Eigen3::Eigen
)

if(NOT ${PROJECT_NAME}_CLANG_FORMAT_BINARY)
    find_program(${PROJECT_NAME}_CLANG_FORMAT_BINARY clang-format)
endif()


if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
add_custom_target(clang-format
        COMMAND ${${PROJECT_NAME}_CLANG_FORMAT_BINARY}
        -i ${ALL_SOURCE_FILES} ${headers} ${test_sources})
endif()

enable_testing()
add_subdirectory(test)

if(${PROJECT_NAME}_ENABLE_DOXYGEN)
    set(DOXYGEN_CALLER_GRAPH YES)
    set(DOXYGEN_CALL_GRAPH YES)
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_EXTRACT_PRIVATE YES)
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs)

    find_package(Doxygen REQUIRED dot)
    doxygen_add_docs(doxygen-docs ${PROJECT_SOURCE_DIR})

    message( VERBOSE "Doxygen has been setup and documentation is now available.")
endif()











