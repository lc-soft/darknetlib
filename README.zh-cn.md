# DarknetLib

[darknet](http://pjreddie.com/darknet/) 的 C API 库

(**中文**/[EN](README.md))

## 特性

- 支持在训练检测器时获取训练进度
- 支持编译成动态库
- 重新定义了接口，接口命名风格参考自 [LeveDB/c.h](https://github.com/google/leveldb/blob/master/include/leveldb/c.h)

## 构建

**在 Windows 下构建**

1. 下载安装 [CUDA](https://developer.nvidia.com/cuda-downloads)
1. 下载 [cuDNN](https://developer.nvidia.com/cudnn) 并解压到 `3rdparty` 目录
1. 使用 Visual Studio 打开 `build/darknet.sln`
1. 设置配置为 **Release** 和 **x64**
1. 构建 darknet_gpu 和 test 项目

## 运行

1. 下载 [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) 文件至 `test` 目录
1. 在 Visual Studio 中按 F5 键运行测试程序

## 许可

代码基于 [MIT 许可协议](LICENSE) 发布。
