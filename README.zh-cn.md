# DarknetLib

[darknet](http://pjreddie.com/darknet/) 的 C API 库

(**中文**/[EN](README.md))

## 特性

- 支持简单的异常处理
- 重新定义了接口，接口命名风格参考自 [LeveDB/c.h](https://github.com/google/leveldb/blob/master/include/leveldb/c.h)
- 适合作为动态库来使用

## 构建

**在 Windows 中构建**

1. 下载安装 [CUDA](https://developer.nvidia.com/cuda-downloads)
1. 下载 [cuDNN](https://developer.nvidia.com/cudnn) 并解压到 `3rdparty` 目录
1. 使用 Visual Studio 打开 `build/darknet.sln`
1. 设置配置为 **Release** 和 **x64**
1. 构建 darknet_gpu 和 test 项目

**在 Linux 中构建**

暂未添加相关构建脚本，你可以参考 [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) 中的 Makefile 文件来尝试编写构建脚本。如果你对解决这方面的问题很熟练，可以向我们提交拉取请求（Pull Reqeust）。

## 运行

1. 下载 [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) 文件至 `test` 目录
1. 在 Visual Studio 中按 F5 键运行测试程序

## 许可

代码基于 [MIT 许可协议](LICENSE) 发布。
