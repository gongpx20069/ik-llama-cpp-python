cmake_minimum_required(VERSION 3.21)
project(ik_llama_cpp_python LANGUAGES C CXX)

# Build ik_llama.cpp from vendor submodule
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

add_subdirectory(vendor/ik_llama.cpp)

# Install shared libraries into the Python package
install(
    TARGETS llama ggml ggml-base ggml-cpu
    LIBRARY DESTINATION ik_llama_cpp/lib
    RUNTIME DESTINATION ik_llama_cpp/lib
)
