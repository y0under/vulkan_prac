cmake_minimum_required(VERSION 3.10)

# プロジェクト名とC++の標準
project(VulkanTest VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 最適化オプション
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# Vulkanのパッケージを見つける
find_package(Vulkan REQUIRED)

# MoltenVKのパスを指定
set(MOLTENVK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/MoltenVK")

# MoltenVKのヘッダーとライブラリのパスを追加
include_directories("${MOLTENVK_PATH}/include")
link_directories("${MOLTENVK_PATH}/lib")

# ソースファイル
add_executable(VulkanTest main.cc)

# リンカオプション
target_link_libraries(VulkanTest
    Vulkan::Vulkan      # Vulkanライブラリ
    glfw                # GLFWライブラリ
    dl                  # dlライブラリ
    pthread             # pthreadライブラリ
    X11                 # X11ライブラリ
    Xxf86vm             # Xxf86vmライブラリ
    Xrandr              # Xrandrライブラリ
    Xi                  # Xiライブラリ
    MoltenVK            # MoltenVKライブラリ
)

# シェーダーのパスを設定
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shader")

# シェーダーをビルドディレクトリにコピーするカスタムコマンドを設定
add_custom_command(
    TARGET VulkanTest PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${SHADER_DIR}" "$<TARGET_FILE_DIR:VulkanTest>/shader"
)

# RPATHの設定
set_target_properties(VulkanTest PROPERTIES
    INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/MoltenVK/lib" # MoltenVKのライブラリパス
    BUILD_WITH_INSTALL_RPATH TRUE
)

# CMakeでのインストール時にRPATHを設定
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

