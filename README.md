# DarknetLib

([中文](README.zh-cn.md)/**EN**)

## Tab of contents

- [Introduction](#introduction)
- [Features](#features)
- [Use](#use)
    - [Use on Windows](#use-on-windows)
    - [Use on Linux](#use-on-linux)
- [Todo](#todo)
- [Legal](#legal)

## Introduction

Darknetlib is a C library for wrapping [darknet](http://pjreddie.com/darknet/),
It mainly provides object detection support for [LC-Finder](https://github.com/lc-soft/LC-Finder), you can find the relevant code in the [src/lib/detector.c](https://github.com/lc-soft/LC-Finder/blob/develop/src/lib/detector.c) file.

## Features

- Provides simple exception handling
- Redefining the interface with a new naming style
- Suitable for compilation into a dynamic library to use

## Use

### Use on Windows

**Build:**

1. Download and install [CUDA](https://developer.nvidia.com/cuda-downloads)
1. Download [cuDNN](https://developer.nvidia.com/cudnn) and unpack files to hte `3rdparty` directory
1. Open `build/darknet.sln` whith Visual Studio 2017
1. Set **Release** and **x64**
1. build **darknet_gpu** and **test** project

**Run:**

1. Download the [yolov3-tiny.weights](https://pjreddie.com/media/files/yolov3-tiny.weights) file to the `test` directory
1. Run the test

### Use on Linux

```bash
git clone https://github.com/lc-soft/darknetlib.git
git submodule init
git submodule update
make
cd test
make
```

## Todo

- Let `darknet_detector_train()` work fine.
- Friendly and standardized log output.
- Remove command line interaction code, like: `getchar()`.
- Remove the strange `system()` call, like: `system("echo ...")`
- Refactor source code of the darknet, make it more suitable for use as a library, not a command line program. (very low priority)

## Legal

Code released under the [MIT License](LICENSE).
