# MITSUBA 渲染

## 安装 MITSUBA 源码并编译
官方参考教程：https://mitsuba.readthedocs.io/en/latest/src/developer_guide/compiling.html#windows

本文档演示所用系统：Windows11, Visual Studio 2022, cmake 3.28.4, Python 3.10.1

### 安装 CMAKE
https://cmake.org/download/

下载合适版本的 cmake (.msi)文件进行安装

(https://github.com/Kitware/CMake/releases/download/v3.29.0/cmake-3.29.0-windows-x86_64.msi)

### 下载 MITSUAB 源码
`git clone --recursive https://github.com/mitsuba-renderer/mitsuba3`

进入 mitsuba3 目录下

`cmake -G "Visual Studio 17 2022" -A x64 -B build`

进入生成的 `build` 文件夹，打开名为 `mitsuba.conf` 的文件，在第 87 行加上 `"scalar_spectral_polarized", "cuda_spectral_polarized"`，保存并返回 mitsuba3 目录下

`cmake --build build --config Release`

这个过程时间可能会比较长

完成 mitsuba3 源码的编译后，进入 `build\Release` 目录，运行 `setpath`

至此完成对 mitsuba3 的安装。
