# DarknetLib

C bindings for [darknet](http://pjreddie.com/darknet/).

([中文](README.zh-cn.md)/[EN](README.md))

## Example

``` c
#include "darknet.h"

int main(int argc, char *argv[])
{
	darknet_network_t *net;
	darknet_detector_t *d;
	darknet_config_t *cfg;
	darknet_dataconfig_t *datacfg;
	darknet_detections_t dets;

    cfg = darknet_config_create("cfg/yolov3.cfg");
	datacfg = darknet_dataconfig_create("cfg/coco.data");
	net = darknet_network_create(cfg);
	darknet_network_load_weights(net, "yolov3.weights");
	d = darnet_detector_create(net, datacfg);
	if (d) {
		darknet_detector_test(d, "img/dog.jpg", &dets);
		// do somethings
		// ...
		darknet_detections_destroy(&dets);
		darknet_detector_destroy(d);
	}
	darknet_config_destroy(cfg);
	darknet_network_destroy(net);
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
1. Open `build/darknet.sln` whith Visual Studio 2017
1. Set **Release** and **x64**
1. build **darknet_gpu** and **test** project

## Run

1. Download the [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) file to the `test` directory
1. Run the test

## Legal

Code released under the [MIT License](LICENSE).
