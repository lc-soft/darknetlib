#include <stdio.h>
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

int main(int argc, char *argv[])
{
	clock_t c;
	darknet_network_t *net;
	darknet_detector_t *d;
	darknet_config_t *cfg;
	darknet_dataconfig_t *datacfg;
	darknet_detections_t dets;

	c = clock();
	cfg = darknet_config_create("cfg/yolov3.cfg");
	datacfg = darknet_dataconfig_create("cfg/coco.data");
	net = darknet_network_create(cfg);
	darknet_network_load_weights(net, "yolov3.weights");
	d = darnet_detector_create(net, datacfg);
	if (d) {
		printf("\ntime: %.2fs\n\n",
		       (clock() - c) * 1.0f / CLOCKS_PER_SEC);
		c = clock();
		darknet_detector_test(d, "img/dog.jpg", &dets);
		printf("\ntime: %.2fs\n\n",
		       (clock() - c) * 1.0f / CLOCKS_PER_SEC);
		print_detections(&dets);
		darknet_detections_destroy(&dets);
		darknet_detector_destroy(d);
	}
	darknet_config_destroy(cfg);
	darknet_dataconfig_destroy(datacfg);
	darknet_network_destroy(net);
	return 0;
}
