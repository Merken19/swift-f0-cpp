cmake_minimum_required(VERSION 3.13)
project(swift_f0_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the path to ONNX Runtime
set(ONNXRUNTIME_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/nuget_packages/Microsoft.ML.OnnxRuntime.1.22.1")
set(ONNXRUNTIME_LIB_PATH "${ONNXRUNTIME_ROOT_PATH}/runtimes/win-x64/native")
set(ONNXRUNTIME_INCLUDE_PATH "${ONNXRUNTIME_ROOT_PATH}/build/native/include")

# Find ONNX Runtime
if(NOT EXISTS ${ONNXRUNTIME_ROOT_PATH})
    message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_ROOT_PATH}")
endif()

# Add executable
add_executable(swift_f0 
    main.cpp
    SwiftF0.cpp
    SwiftF0.h
)

# Include directories
target_include_directories(swift_f0 PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXRUNTIME_INCLUDE_PATH}
)

# Link libraries
target_link_libraries(swift_f0 
    ${ONNXRUNTIME_LIB_PATH}/onnxruntime.lib
)

# Copy ONNX Runtime DLL to output directory
add_custom_command(TARGET swift_f0 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_LIB_PATH}/onnxruntime.dll"
    $<TARGET_FILE_DIR:swift_f0>
)

# Copy model.onnx to output directory
add_custom_command(TARGET swift_f0 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/model.onnx"
    $<TARGET_FILE_DIR:swift_f0>
)