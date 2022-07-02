find_package(Catch2 3 REQUIRED)

add_executable(tests)
target_link_libraries(tests PRIVATE
        Catch2::Catch2WithMain
        doom_lib)

target_include_directories(tests PRIVATE
        ${doom_SOURCE_DIR}/linuxdoom-1.10)

set_target_properties(tests PROPERTIES CUDA_ARCHITECTURES "86")

add_subdirectory(geometry)