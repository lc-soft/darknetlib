# DarknetLib

(**中文**/[EN](README.md))

## 目录

- [介绍](#介绍)
- [特性](#特性)
- [使用](#使用)
    - [在 Windows 中使用](#在-windows-中使用)
    - [在 Linux 中使用](#在-linux-中使用)
- [计划](#计划)
- [许可](#许可)

## 介绍

Darknetlib 是 [darknet](http://pjreddie.com/darknet/) 的 C API 库，主要为 [LC-Finder](https://github.com/lc-soft/LC-Finder) 提供目标检测功能，相关代码可参考它的 [src/lib/detector.c](https://github.com/lc-soft/LC-Finder/blob/develop/src/lib/detector.c) 文件。

## 特性

- 支持简单的异常处理
- 重新定义了接口，接口命名风格参考自 [LeveDB/c.h](https://github.com/google/leveldb/blob/master/include/leveldb/c.h)
- 适合作为动态库来使用

## 使用

### 在 Windows 中使用

使用 lcpkg 安装：

```bash
lcpkg install github.com/lc-soft/darknetlib
```

这种方式安装的是纯 CPU 运算的版本，如果你需要带 GPU 加速的版本，请前往[发行版页面](https://github.com/lc-soft/darknetlib/releases)手动下载。

如果你想手动从源码构建的话：

1. 下载安装 [CUDA](https://developer.nvidia.com/cuda-downloads)
1. 下载 [cuDNN](https://developer.nvidia.com/cudnn) 并解压到 `3rdparty` 目录
1. 使用 Visual Studio 打开 `build/darknet.sln`
1. 设置配置为 **Release** 和 **x64**
1. 构建 darknet_gpu 和 test 项目
1. 下载 [yolov3-tiny.weights](https://pjreddie.com/media/files/yolov3-tiny.weights) 文件至 `test` 目录
1. 在 Visual Studio 中按 F5 键运行测试程序

### 在 Linux 中使用

```bash
git clone https://github.com/lc-soft/darknetlib.git
git submodule init
git submodule update
make
cd test
make
```

## 计划

- 让 `darknet_detector_train()` 能够正常训练识别器。
- 规范化输出的日志格式。
- 移除命令行交互代码，例如：`getchar()`。
- 移除奇怪的 `system()` 调用，例如：`system("echo ...")`。
- 重构 darknet，让它更适合作为函数库使用，而不是命令行程序。（极低优先级）

## 许可

代码基于 [MIT 许可协议](LICENSE) 发布。
