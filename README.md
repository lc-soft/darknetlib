# DarknetLib

C bindings for [darknet](http://pjreddie.com/darknet/).

([中文](README.zh-cn.md)/[EN](README.md))

## Example

``` c
#include "darknet.h"

int main(int argc, char *argv[])
{
    darknet_detector_t *d;
    darknet_detections_t dets;

    d = darnet_detector_create("coco.data", "yolov3.cfg", "yolov3.weights");
    darknet_detector_test(d, "img/dog.jpg", &dets);
    // do somethings
    // ...
    darknet_detections_destroy(&dets);
    darknet_detector_destroy(d);
    return 0;
}
```

## Features

- Support for get training progress while training the detector
- Support for compiling into dynamic library
- Redefining the interface with a new naming style

## Build

**Build on Windows**

1. Download and install [CUDA](https://developer.nvidia.com/cuda-downloads)
1. Download [cuDNN](https://developer.nvidia.com/cudnn) and unpack files to hte `3rdparty` directory
1. Open `build/darknet.sh` whith Visual Studio 2017
1. Set **Release** and **x64**
1. build **darknet_gpu** and **test** project

## Run

1. Download the [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) file to the `test` directory
1. Run the test

## Legal

Code released under the [MIT License](LICENSE).