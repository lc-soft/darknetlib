#include <errno.h>
#include "build.h"
#include "darknet.h"
#include "../darknet/src/list.h"
#include "../darknet/src/utils.h"
#include "../darknet/src/option_list.h"
#include "../darknet/src/network.h"
#include "../darknet/src/parser.h"

int check_mistakes = 0;

typedef image darknet_image;
typedef layer darknet_layer;
typedef network darknet_network;
typedef detection darknet_detection;

struct darknet_detector {
	char **names;
	float thresh;
	float hier_thresh;
	darknet_network net;
};

darknet_detector_t *darnet_detector_create(const char *datacfg,
					   const char *cfgfile,
					   const char *weightfile)
{
	darknet_detector_t *detector;
	darknet_network net = parse_network_cfg_custom((char *)cfgfile, 1);
	list *options = read_data_cfg((char *)datacfg);

	int names_size = 0;
	char *names_file = option_find_str(options, "names", "data/names.list");
	char **names = get_labels_custom(names_file, &names_size);

	if (weightfile) {
		load_weights(&net, (char *)weightfile);
	}
	fuse_conv_batchnorm(net);
	calculate_binary_weights(net);
	if (net.layers[net.n - 1].classes != names_size) {
		DEBUG_MSG("error: in the file %s number of names %d "
			  "that isn't equal to classes=%d in the file %s \n",
			  names_file, names_size, net.layers[net.n - 1].classes,
			  cfgfile);
	}
	free_list_contents_kvp(options);
	free_list(options);

	detector = malloc(sizeof(struct darknet_detector));
	detector->net = net;
	detector->thresh = 0.24;
	detector->hier_thresh = 0.5;
	detector->names = names;
	return detector;
}

void darknet_detector_destroy(darknet_detector_t *d)
{
	free_ptrs(d->names, d->net.layers[d->net.n - 1].classes);
	free_network(d->net);
}

static void convert_box(darknet_box_t *dst, box *src)
{
	dst->x = src->x;
	dst->y = src->y;
	dst->w = src->w;
	dst->h = src->h;
}

static int convert_detections(const darknet_detector_t *d,
			      const darknet_layer *layer,
			      darknet_detection *raw_dets, int n_dets,
			      darknet_detections_t *dets)
{
	int i, j, k, n, best_class;

	darknet_detection_t *det;
	detection_with_class *selected;

	selected = get_actual_detections(raw_dets, n_dets, d->thresh, &n);
	dets->list = calloc(n, sizeof(darknet_detection_t));
	dets->length = n;
	if (!dets->list) {
		return -ENOMEM;
	}
	for (det = dets->list, i = 0; i < n; ++i, ++det) {
		best_class = selected[i].best_class;
		det->name = d->names[best_class];
		det->names_count = 0;
		convert_box(&det->box, &selected[i].det.bbox);
		for (j = 0; j < layer->classes; ++j) {
			if (selected[i].det.prob[j] > d->thresh &&
			    j != best_class) {
				++det->names_count;
			}
		}
		det->names = malloc(sizeof(char *) * (det->names_count + 1));
		if (det->names) {
			return -ENOMEM;
		}
		for (j = 0, k = 0; j < layer->classes; ++j) {
			if (selected[i].det.prob[j] > d->thresh &&
			    j != best_class) {
				++det->names_count;
				det->names[k++] = d->names[j];
			}
		}
		det->names[det->names_count] = NULL;
	}
	return 0;
}

void darknet_detections_destroy(darknet_detections_t *d)
{
	size_t i;

	for (i = 0; i < d->length; ++i) {
		if (d->list[i].names) {
			free(d->list[i].names);
		}
		d->list[i].name = NULL;
		d->list[i].names = NULL;
	}
	if (d->list) {
		free(d->list);
	}
	d->list = NULL;
	d->length = 0;
}

int darknet_detector_test(darknet_detector_t *d, const char *file,
			  darknet_detections_t *results)
{
	int ret;
	int n_dets = 0;
	float nms = .45;

	darknet_detection *dets;
	darknet_network *net = &d->net;
	darknet_image img = load_image((char *)file, 0, 0, net->c);
	darknet_image sized_img = resize_image(img, net->w, net->h);
	darknet_layer layer = net->layers[net->n - 1];

	network_predict(*net, sized_img.data);
	dets = get_network_boxes(net, img.w, img.h, d->thresh, d->hier_thresh,
				 0, 1, &n_dets, 0);
	if (nms) {
		do_nms_sort(dets, n_dets, layer.classes, nms);
	}
	ret = convert_detections(d, &layer, dets, n_dets, results);
	if (ret != 0) {
		darknet_detections_destroy(results);
		return ret;
	}
	free_image(img);
	free_image(sized_img);
	free_detections(dets, n_dets);
	return 0;
}
