#include <stdio.h>
#include <string.h>
#include <time.h>

#include "../include/darknet.h"

void print_detections(const darknet_detections_t *dets)
{
	size_t i, j;
	darknet_detection_t *det;

	printf("detections:\n");
	for (i = 0; i < dets->length; ++i) {
		det = &dets->list[i];
		printf("[%zu] best name: %s, box: (%g, %g, %g, %g)\n", i,
		       det->best_name, det->box.x, det->box.y, det->box.w,
		       det->box.h);
		printf("names: \n");
		for (j = 0; j < det->names_count; ++j) {
			printf("%s, %g%%\n", det->names[j],
			       det->prob[j] * 100.0f);
		}
	}
}

int detect(void)
{
	clock_t c;
	int code = 0;

	// initialize variables so that darknet can check if they need to be
	// destroyed after catching exception
	darknet_config_t *cfg = NULL;
	darknet_dataconfig_t *datacfg = NULL;
	darknet_detections_t dets = { 0 };
	darknet_detector_t *d = NULL;
	darknet_network_t *net = NULL;

	c = clock();
	darknet_try
	{
		cfg = darknet_config_load("cfg/yolov3.cfg");
		datacfg = darknet_dataconfig_load("cfg/coco.data");

		net = darknet_network_create(cfg);
		darknet_network_load_weights(net, "yolov3.weights");
		d = darknet_detector_create(net, datacfg);

		printf("\ntime: %.2fs\n\n",
		       (clock() - c) * 1.0f / CLOCKS_PER_SEC);
		c = clock();
		darknet_detector_test(d, "img/dog.jpg", &dets);
		printf("\ntime: %.2fs\n\n",
		       (clock() - c) * 1.0f / CLOCKS_PER_SEC);
	}
	darknet_catch(err)
	{
		printf("error: %s\n", darknet_get_last_error_string());
		code = -1;
	}
	darknet_etry;

	print_detections(&dets);

	darknet_detections_destroy(&dets);
	darknet_config_destroy(cfg);
	darknet_dataconfig_destroy(datacfg);
	darknet_detector_destroy(d);
	darknet_network_destroy(net);
	return code;
}

int train(void)
{
	int code = 0;
	darknet_config_t *cfg = NULL;
	darknet_dataconfig_t *datacfg = NULL;
	darknet_detector_t *d = NULL;
	darknet_network_t *net = NULL;

	darknet_try
	{
		cfg = darknet_config_load("train/yieldsign.cfg");
		datacfg = darknet_dataconfig_load("train/yieldsign.data");
		// Reset wrokdir to the train
		darknet_config_set_workdir(cfg, "train");
		darknet_dataconfig_set_workdir(datacfg, "train");

		net = darknet_network_create(cfg);
		darknet_network_load_weights(net, "train/darknet53.conv.74");
		d = darknet_detector_create(net, datacfg);
		darknet_detector_train(d);
	}
	darknet_catch(err)
	{
		printf("error: %s\n", darknet_get_last_error_string());
		code = -1;
	}
	darknet_etry;

	darknet_config_destroy(cfg);
	darknet_dataconfig_destroy(datacfg);
	darknet_detector_destroy(d);
	darknet_network_destroy(net);
	return code;
}

int main(int argc, char *argv[])
{
	if (argc > 1) {
		if (strcmp(argv[1], "detect") == 0) {
			return detect();
		} else if (strcmp(argv[1], "train") == 0) {
			return train();
		} else {
			return -1;
		}
	}
	return detect();
}
