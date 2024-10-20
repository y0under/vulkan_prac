cmake_minimum_required(VERSION 3.10)

# プロジェクト名とC++の標準
project(VulkanTest VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 最適化オプション
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# Vulkanのパッケージを見つける
find_package(Vulkan REQUIRED)

add_executable(VulkanTest main.cc)

target_link_libraries(VulkanTest
                      Vulkan::Vulkan
                      glfw
                      dl
                      pthread
                      X11
                      Xxf86vm
                      Xrandr
                      Xi
                      MoltenVK
)

set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shader")

add_custom_command(
    TARGET VulkanTest PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${SHADER_DIR}" "$<TARGET_FILE_DIR:VulkanTest>/shader"
)

# my engine
include(FetchContent)
FetchContent_Declare(
  y0_engine
  GIT_REPOSITORY https://github.com/y0under/y0_engine.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(y0_engine)

include_directories(${CMAKE_BINARY_DIR}/_deps/y0_engine-src/include)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

