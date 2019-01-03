#include <errno.h>
#include "build.h"
#include "darknet.h"
#include "../darknet/src/list.h"
#include "../darknet/src/utils.h"
#include "../darknet/src/option_list.h"
#include "../darknet/src/network.h"
#include "../darknet/src/parser.h"

#ifdef _WIN32
#define PATH_SEP '\\'
#else
#define PATH_SEP '/'
#endif

int check_mistakes = 0;

typedef image darknet_image;
typedef layer darknet_layer;
typedef network darknet_network;
typedef detection darknet_detection;

struct darknet_config {
	list *options;
};

struct darknet_network {
	network net;
};

struct darknet_detector {
	char **names;
	float thresh;
	float hier_thresh;
	darknet_network_t *net;
};

darknet_config_t *darknet_config_create(const char *file)
{
	darknet_config_t *cfg;

	cfg = malloc(sizeof(darknet_config_t));
	if (!cfg) {
		return NULL;
	}
	cfg->options = read_data_cfg((char *)file);
	return cfg;
}

void darknet_config_destroy(darknet_config_t *cfg)
{
	free_list_contents_kvp(cfg->options);
	free_list(cfg->options);
	free(cfg);
}

static int check_key_is_path(const char *key)
{
	size_t i;
	const char *paths[6] = { "train",  "valid",  "names",
				 "labels", "backup", "map" };
	for (i = 0; i < 6; ++i) {
		if (strcmp(paths[i], key) == 0) {
			return 0;
		}
	}
	return -1;
}

size_t darknet_config_set_workdir(darknet_config_t *cfg, const char *workdir)
{
	kvp *kv;
	node *next;
	char *key;
	char *val;

	size_t len;
	size_t key_len;
	size_t dir_len = strlen(workdir);
	size_t count = 0;

	for (next = cfg->options->front; next; next = next->next) {
		kv = next->val;
		if (check_key_is_path(kv->key) != 0) {
			continue;
		}
		if (!kv->val || kv->val[0] == '/' || kv->val[0] == '\\') {
			continue;
		}
		key = kv->key;
		key_len = strlen(kv->key);
		len = strlen(kv->val) + key_len + dir_len + 2;
		val = malloc(sizeof(char) * (len + 1));
		sprintf(val, "%s-%s%c%s", key, workdir, PATH_SEP, kv->val);
		val[key_len] = 0;
		val[len] = 0;
		kv->key = val;
		kv->val = val + key_len + 1;
		free(key);
		++count;
	}
	return count;
}

darknet_network_t *darknet_network_create(const char *cfgfile)
{
	darknet_network_t *net;

	net = malloc(sizeof(darknet_network_t));
	if (!net) {
		return NULL;
	}
	net->net = parse_network_cfg_custom((char *)cfgfile, 1);
	return net;
}

void darknet_network_destroy(darknet_network_t *net)
{
	free_network(net->net);
	free(net);
}

void darknet_network_load_weights(darknet_network_t *net,
				  const char *weightfile)
{
	load_weights(&net->net, (char *)weightfile);
	fuse_conv_batchnorm(net->net);
	calculate_binary_weights(net->net);
}

static list *get_paths(char *filename)
{
	char *path;
	list *lines;
	FILE *file = fopen(filename, "r");

	if (!file) {
		return NULL;
	}
	lines = make_list();
	while ((path = fgetl(file))) {
		list_insert(lines, path);
	}
	fclose(file);
	return lines;
}

static char **get_labels_custom(char *filename, int *size)
{
	char **labels;
	list *plist = get_paths(filename);

	if (!plist) {
		return NULL;
	}
	if (size) {
		*size = plist->size;
	}
	labels = (char **)list_to_array(plist);
	free_list(plist);
	return labels;
}

darknet_detector_t *darnet_detector_create(darknet_network_t *net,
					   darknet_config_t *cfg)
{
	darknet_detector_t *detector;
	list *options = cfg->options;

	int names_size = 0;
	char *names_file = option_find_str(options, "names", "data/names.list");
	char **names = get_labels_custom(names_file, &names_size);

	if (!names) {
		DEBUG_MSG("error: couldn't open file: %s\n", names_file);
		return NULL;
	}
	if (net->net.layers[net->net.n - 1].classes != names_size) {
		DEBUG_MSG("error: in the file %s number of names %d "
			  "that isn't equal to classes=%d\n",
			  names_file, names_size,
			  net->net.layers[net->net.n - 1].classes);
	}

	detector = malloc(sizeof(struct darknet_detector));
	detector->net = net;
	detector->thresh = 0.24;
	detector->hier_thresh = 0.5;
	detector->names = names;
	return detector;
}

void darknet_detector_destroy(darknet_detector_t *d)
{
	free_ptrs(d->names, d->net->net.layers[d->net->net.n - 1].classes);
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
		det->best_name = d->names[best_class];
		det->names_count = 0;
		convert_box(&det->box, &selected[i].det.bbox);
		for (j = 0; j < layer->classes; ++j) {
			if (selected[i].det.prob[j] > d->thresh) {
				++det->names_count;
			}
		}
		det->names = malloc(sizeof(char *) * (det->names_count + 1));
		det->prob = malloc(sizeof(float) * (det->names_count));
		if (!det->names || !det->prob) {
			return -ENOMEM;
		}
		for (j = 0, k = 0; j < layer->classes; ++j) {
			if (selected[i].det.prob[j] > d->thresh) {
				det->names[k] = d->names[j];
				det->prob[k] = selected[i].det.prob[j];
				k++;
			}
		}
		det->names[det->names_count] = NULL;
	}
	return 0;
}

void darknet_detections_destroy(darknet_detections_t *dets)
{
	size_t i;
	darknet_detection_t *det;

	for (i = 0; i < dets->length; ++i) {
		det = &dets->list[i];
		if (det->names) {
			free(det->names);
		}
		if (det->prob) {
			free(det->prob);
		}
		det->best_name = NULL;
		det->names = NULL;
	}
	if (dets->list) {
		free(dets->list);
	}
	dets->list = NULL;
	dets->length = 0;
}

int darknet_detector_test(darknet_detector_t *d, const char *file,
			  darknet_detections_t *results)
{
	int ret;
	int n_dets = 0;
	float nms = .45;

	darknet_detection *dets;
	darknet_network *net = &d->net->net;
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
