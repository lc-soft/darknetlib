#include <errno.h>
#include "build.h"
#include "darknet.h"
#include "../darknet/include/darknet.h"
#include "../darknet/src/list.h"
#include "../darknet/src/utils.h"
#include "../darknet/src/option_list.h"
#include "../darknet/src/network.h"
#include "../darknet/src/parser.h"
#include "../darknet/src/parser.c"
#include "../include/darknet.h"

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
	list *sections;
};

struct darknet_dataconfig {
	list *options;
	char *name;
};

struct darknet_network {
	char *file;
	network net;
	darknet_config_t *cfg;
};

struct darknet_detector {
	char **names;
	float thresh;
	float hier_thresh;
	darknet_network_t *net;
	darknet_dataconfig_t *datacfg;
};

static int check_key_is_path(const char *key)
{
	size_t i;
	const char *paths[7] = { "train",  "valid", "names", "labels",
				 "backup", "map",   "tree" };
	for (i = 0; i < 7; ++i) {
		if (strcmp(paths[i], key) == 0) {
			return 0;
		}
	}
	return -1;
}

static size_t set_options_workdir(list *options, const char *workdir)
{
	kvp *kv;
	node *next;
	char *key;
	char *val;

	size_t len;
	size_t key_len;
	size_t dir_len = strlen(workdir);
	size_t count = 0;

	for (next = options->front; next; next = next->next) {
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

darknet_config_t *darknet_config_load(const char *file)
{
	darknet_config_t *cfg;

	cfg = malloc(sizeof(darknet_config_t));
	if (!cfg) {
		return NULL;
	}
	cfg->sections = read_cfg((char *)file);
	return cfg;
}

void darknet_config_destroy(darknet_config_t *cfg)
{
	node *next;

	if (!cfg) {
		return;
	}
	for (next = cfg->sections->front; next; next = next->next) {
		free_section(next->val);
	}
	free_list(cfg->sections);
	free(cfg);
}

darknet_dataconfig_t *darknet_dataconfig_load(const char *file)
{
	darknet_dataconfig_t *cfg;

	cfg = malloc(sizeof(darknet_dataconfig_t));
	if (!cfg) {
		return NULL;
	}
	cfg->options = read_data_cfg((char *)file);
	cfg->name = basecfg((char *)file);
	return cfg;
}

int darknet_dataconfig_get_classes(darknet_dataconfig_t *cfg)
{
	return option_find_int_quiet(cfg->options, "classes", 0);
}

void darknet_dataconfig_destroy(darknet_dataconfig_t *cfg)
{
	if (!cfg) {
		return;
	}
	free_list_contents_kvp(cfg->options);
	free_list(cfg->options);
	free(cfg->name);
	free(cfg);
}

size_t darknet_dataconfig_set_workdir(darknet_dataconfig_t *cfg,
				      const char *workdir)
{
	return set_options_workdir(cfg->options, workdir);
}

size_t darknet_config_set_workdir(darknet_config_t *cfg, const char *workdir)
{
	node *next;
	section *current;
	size_t count = 0;

	for (next = cfg->sections->front; next; next = next->next) {
		current = next->val;
		if (current->type && strcmp(current->type, "[region]") == 0) {
			count += set_options_workdir(current->options, workdir);
		}
	}
	return count;
}

static int load_weights_file(network *net, const char *filename)
{
	int i;
	int major;
	int minor;
	int revision;
	int cutoff = net->n;
	FILE *fp;

	DEBUG_MSG("Loading weights from %s...\n", filename);
#ifdef GPU
	if (net->gpu_index >= 0) {
		cuda_set_device(net->gpu_index);
	}
#endif
	fp = fopen(filename, "rb");
	if (!fp) {
		DEBUG_ERROR("Cannot open file: %s\n", filename);
		return -ENOENT;
	}

	fread(&major, sizeof(int), 1, fp);
	fread(&minor, sizeof(int), 1, fp);
	fread(&revision, sizeof(int), 1, fp);
	if ((major * 10 + minor) >= 2) {
		printf("\n seen 64 \n");
		uint64_t iseen = 0;
		fread(&iseen, sizeof(uint64_t), 1, fp);
		*net->seen = iseen;
	} else {
		printf("\n seen 32 \n");
		fread(net->seen, sizeof(int), 1, fp);
	}
	int transpose = (major > 1000) || (minor > 1000);

	for (i = 0; i < net->n && i < cutoff; ++i) {
		layer l = net->layers[i];
		if (l.dontload)
			continue;
		if (l.type == CONVOLUTIONAL) {
			load_convolutional_weights(l, fp);
		}
		if (l.type == CONNECTED) {
			load_connected_weights(l, fp, transpose);
		}
		if (l.type == BATCHNORM) {
			load_batchnorm_weights(l, fp);
		}
		if (l.type == CRNN) {
			load_convolutional_weights(*(l.input_layer), fp);
			load_convolutional_weights(*(l.self_layer), fp);
			load_convolutional_weights(*(l.output_layer), fp);
		}
		if (l.type == RNN) {
			load_connected_weights(*(l.input_layer), fp, transpose);
			load_connected_weights(*(l.self_layer), fp, transpose);
			load_connected_weights(*(l.output_layer), fp,
					       transpose);
		}
		if (l.type == GRU) {
			load_connected_weights(*(l.input_z_layer), fp,
					       transpose);
			load_connected_weights(*(l.input_r_layer), fp,
					       transpose);
			load_connected_weights(*(l.input_h_layer), fp,
					       transpose);
			load_connected_weights(*(l.state_z_layer), fp,
					       transpose);
			load_connected_weights(*(l.state_r_layer), fp,
					       transpose);
			load_connected_weights(*(l.state_h_layer), fp,
					       transpose);
		}
		if (l.type == LOCAL) {
			int locations = l.out_w * l.out_h;
			int size = l.size * l.size * l.c * l.n * locations;
			fread(l.biases, sizeof(float), l.outputs, fp);
			fread(l.weights, sizeof(float), size, fp);
#ifdef GPU
			if (gpu_index >= 0) {
				push_local_layer(l);
			}
#endif
		}
	}
	DEBUG_MSG("Done!\n");
	fclose(fp);
	return 0;
}

// Modified from darknet\src\parser.c:710
static int init_network(network *out_net, darknet_config_t *cfg, int batch)
{
	section *s;
	network net;
	size_params params;
	list *options;
	list *sections = cfg->sections;
	node *n = sections->front;

	if (!n) {
		DEBUG_ERROR("Config file has no sections");
		return -1;
	}
	net = make_network(sections->size - 1);
	net.gpu_index = gpu_index;
	s = (section *)n->val;
	options = s->options;
	if (!is_network(s)) {
		DEBUG_ERROR("First section must be [net] or [network]");
		return -2;
	}
	parse_net_options(options, &net);

	params.h = net.h;
	params.w = net.w;
	params.c = net.c;
	params.inputs = net.inputs;
	if (batch > 0)
		net.batch = batch;
	if (net.batch < net.time_steps)
		net.batch = net.time_steps;
	params.batch = net.batch;
	params.time_steps = net.time_steps;
	params.net = net;

	int count = 0;
	float bflops = 0;
	size_t max_inputs = 0;
	size_t max_outputs = 0;
	size_t workspace_size = 0;

	n = n->next;
	fprintf(stderr, "layer     filters    size              input          "
			"      output\n");
	while (n) {
		params.index = count;
		fprintf(stderr, "%4d ", count);
		s = (section *)n->val;
		options = s->options;
		layer l = { (LAYER_TYPE)0 };
		LAYER_TYPE lt = string_to_layer_type(s->type);
		if (lt == CONVOLUTIONAL) {
			l = parse_convolutional(options, params);
		} else if (lt == LOCAL) {
			l = parse_local(options, params);
		} else if (lt == ACTIVE) {
			l = parse_activation(options, params);
		} else if (lt == RNN) {
			l = parse_rnn(options, params);
		} else if (lt == GRU) {
			l = parse_gru(options, params);
		} else if (lt == LSTM) {
			l = parse_lstm(options, params);
		} else if (lt == CRNN) {
			l = parse_crnn(options, params);
		} else if (lt == CONNECTED) {
			l = parse_connected(options, params);
		} else if (lt == CROP) {
			l = parse_crop(options, params);
		} else if (lt == COST) {
			l = parse_cost(options, params);
		} else if (lt == REGION) {
			l = parse_region(options, params);
		} else if (lt == YOLO) {
			l = parse_yolo(options, params);
		} else if (lt == DETECTION) {
			l = parse_detection(options, params);
		} else if (lt == SOFTMAX) {
			l = parse_softmax(options, params);
			net.hierarchy = l.softmax_tree;
		} else if (lt == NORMALIZATION) {
			l = parse_normalization(options, params);
		} else if (lt == BATCHNORM) {
			l = parse_batchnorm(options, params);
		} else if (lt == MAXPOOL) {
			l = parse_maxpool(options, params);
		} else if (lt == REORG) {
			l = parse_reorg(options, params);
		} else if (lt == REORG_OLD) {
			l = parse_reorg_old(options, params);
		} else if (lt == AVGPOOL) {
			l = parse_avgpool(options, params);
		} else if (lt == ROUTE) {
			l = parse_route(options, params, net);
			int k;
			for (k = 0; k < l.n; ++k)
				net.layers[l.input_layers[k]].use_bin_output =
				    0;
		} else if (lt == UPSAMPLE) {
			l = parse_upsample(options, params, net);
		} else if (lt == SHORTCUT) {
			l = parse_shortcut(options, params, net);
			net.layers[count - 1].use_bin_output = 0;
			net.layers[l.index].use_bin_output = 0;
		} else if (lt == DROPOUT) {
			l = parse_dropout(options, params);
			l.output = net.layers[count - 1].output;
			l.delta = net.layers[count - 1].delta;
#ifdef GPU
			l.output_gpu = net.layers[count - 1].output_gpu;
			l.delta_gpu = net.layers[count - 1].delta_gpu;
#endif
		} else {
			DEBUG_MSG("Type not recognized: %s\n", s->type);
		}
		l.onlyforward =
		    option_find_int_quiet(options, "onlyforward", 0);
		l.stopbackward =
		    option_find_int_quiet(options, "stopbackward", 0);
		l.dontload = option_find_int_quiet(options, "dontload", 0);
		l.dontloadscales =
		    option_find_int_quiet(options, "dontloadscales", 0);
		l.learning_rate_scale =
		    option_find_float_quiet(options, "learning_rate", 1);
		option_unused(options);
		net.layers[count] = l;
		if (l.workspace_size > workspace_size) {
			workspace_size = l.workspace_size;
		}
		if (l.inputs > max_inputs) {
			max_inputs = l.inputs;
		}
		if (l.outputs > max_outputs) {
			max_outputs = l.outputs;
		}
		n = n->next;
		++count;
		if (n) {
			params.h = l.out_h;
			params.w = l.out_w;
			params.c = l.out_c;
			params.inputs = l.outputs;
		}
		if (l.bflops > 0)
			bflops += l.bflops;
	}
	net.outputs = get_network_output_size(net);
	net.output = get_network_output(net);
	fprintf(stderr, "Total BFLOPS %5.3f \n", bflops);
#ifdef GPU
	get_cuda_stream();
	get_cuda_memcpy_stream();
	if (gpu_index >= 0) {
		int size = get_network_input_size(net) * net.batch;
		net.input_state_gpu = cuda_make_array(0, size);
		if (cudaSuccess == cudaHostAlloc(&net.input_pinned_cpu,
						 size * sizeof(float),
						 cudaHostRegisterMapped))
			net.input_pinned_cpu_flag = 1;
		else {
			cudaGetLastError();    // reset CUDA-error
			net.input_pinned_cpu =
			    (float *)calloc(size, sizeof(float));
		}

		// pre-allocate memory for inference on Tensor Cores (fp16)
		if (net.cudnn_half) {
			*net.max_input16_size = max_inputs;
			CHECK_CUDA(
			    cudaMalloc((void **)net.input16_gpu,
				       *net.max_input16_size *
					   sizeof(short)));    // sizeof(half)
			*net.max_output16_size = max_outputs;
			CHECK_CUDA(
			    cudaMalloc((void **)net.output16_gpu,
				       *net.max_output16_size *
					   sizeof(short)));    // sizeof(half)
		}
		if (workspace_size) {
			fprintf(
			    stderr,
			    " Allocate additional workspace_size = %1.2f MB \n",
			    (float)workspace_size / 1000000);
			net.workspace = cuda_make_array(
			    0, workspace_size / sizeof(float) + 1);
		} else {
			net.workspace = (float *)calloc(1, workspace_size);
		}
	}
#else
	if (workspace_size) {
		net.workspace = (float *)calloc(1, workspace_size);
	}
#endif

	LAYER_TYPE lt = net.layers[net.n - 1].type;
	if ((net.w % 32 != 0 || net.h % 32 != 0) &&
	    (lt == YOLO || lt == REGION || lt == DETECTION)) {
		printf("\n Warning: width=%d and height=%d in cfg-file must be "
		       "divisible by 32 for default networks Yolo v1/v2/v3!!! "
		       "\n\n",
		       net.w, net.h);
	}
	*out_net = net;
	return 0;
}

darknet_network_t *darknet_network_create(darknet_config_t *cfg)
{
	darknet_network_t *net;

	net = malloc(sizeof(darknet_network_t));
	if (!net) {
		return NULL;
	}
	net->cfg = cfg;
	if (init_network(&net->net, cfg, 1) == 0) {
		return net;
	}
	free(net);
	return NULL;
}

void darknet_network_destroy(darknet_network_t *net)
{
	if (!net) {
		return;
	}
	free_network(net->net);
	free(net);
}

int darknet_network_load_weights(darknet_network_t *net, const char *weightfile)
{
	int ret;

	ret = load_weights_file(&net->net, (char *)weightfile);
	if (ret != 0) {
		return ret;
	}
	net->file = malloc(sizeof(char) * (strlen(weightfile) + 1));
	strcpy(net->file, weightfile);
	fuse_conv_batchnorm(net->net);
	calculate_binary_weights(net->net);
	return 0;
}

darknet_detector_t *darknet_detector_create(darknet_network_t *net,
					    darknet_dataconfig_t *cfg)
{
	darknet_detector_t *detector;
	list *options = cfg->options;

	int names_size = 0;
	char *names_file = option_find_str(options, "names", "data/names.list");
	char **names = get_labels_custom(names_file, &names_size);

	if (net->net.layers[net->net.n - 1].classes != names_size) {
		darknet_throw(DARKNET_DETECTOR_ERROR,
			      "in the file %s number of names %d "
			      "that isn't equal to classes=%d\n",
			      names_file, names_size,
			      net->net.layers[net->net.n - 1].classes);
	}

	detector = malloc(sizeof(struct darknet_detector));
	detector->net = net;
	detector->thresh = 0.24;
	detector->hier_thresh = 0.5;
	detector->names = names;
	detector->datacfg = cfg;
	return detector;
}

void darknet_detector_destroy(darknet_detector_t *d)
{
	if (!d) {
		return;
	}
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
	int i, j, k, best_class;

	darknet_detection_t *det;

	dets->list = calloc(n_dets, sizeof(darknet_detection_t));
	dets->length = n_dets;
	if (!dets->list) {
		return -ENOMEM;
	}
	for (det = dets->list, i = 0; i < n_dets; ++i) {
		best_class = -1;
		det->best_name = NULL;
		det->names_count = 0;
		convert_box(&det->box, &raw_dets[i].bbox);
		for (j = 0; j < layer->classes; ++j) {
			if (raw_dets[i].prob[j] <= d->thresh) {
				continue;
			}
			if (best_class == -1 ||
			    (raw_dets[i].prob[j] >
			     raw_dets[i].prob[best_class])) {
				best_class = j;
			}
			++det->names_count;
		}
		if (best_class == -1) {
			continue;
		}
		det->best_name = d->names[best_class];
		det->names = malloc(sizeof(char *) * (det->names_count + 1));
		det->prob = malloc(sizeof(float) * (det->names_count));
		if (!det->names || !det->prob) {
			return -ENOMEM;
		}
		for (j = 0, k = 0; j < layer->classes; ++j) {
			if (raw_dets[i].prob[j] > d->thresh) {
				det->names[k] = d->names[j];
				det->prob[k] = raw_dets[i].prob[j];
				k++;
			}
		}
		det->names[det->names_count] = NULL;
		++det;
	}
	dets->length = det - dets->list;
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

// Modified from darknet\src\detector.c:42
void darknet_detector_train(darknet_detector_t *d)
{
	list *options = d->datacfg->options;
	char *train_images =
	    option_find_str(options, "train", "data/train.txt");
	char *valid_images = option_find_str(options, "valid", train_images);
	char *backup_directory = option_find_str(options, "backup", "/backup/");
	int train_images_num = 0;
	float avg_loss = -1;

	int i;
	int gpu = gpu_index;
	int ngpus = 1;
	int *gpus = &gpu;
	int seed = rand();

	network net;
	network *nets = calloc(ngpus, sizeof(network));

	if (ngpus > 1) {
		for (i = 1; i < ngpus; ++i) {
			srand(seed);
#ifdef GPU
			cuda_set_device(gpus[i]);
#endif
			init_network(&nets[i], d->net->cfg, 1);
			load_weights(&nets[i], d->net->file);
			nets[i].learning_rate *= ngpus;
		}
	} else {
		nets[0] = d->net->net;
	}
	net = nets[0];

	int imgs = net.batch * net.subdivisions * ngpus;
	const int actual_batch_size = net.batch * net.subdivisions;

	if (actual_batch_size == 1) {
		darknet_throw(DARKNET_ARG_ERROR,
			      "You set incorrect value batch=1 for Training! "
			      "You should set batch=64 subdivision=64");
	} else if (actual_batch_size < 64) {
		printf("\n Warning: You set batch=%d lower than 64! It is "
		       "recommended to set batch=64 subdivision=64 \n",
		       actual_batch_size);
	}
	printf("Learning Rate: %g, Momentum: %g, Decay: %g\n",
	       net.learning_rate, net.momentum, net.decay);

	data train, buffer;
	layer l = net.layers[net.n - 1];

	int classes = l.classes;
	float jitter = l.jitter;

	list *plist = get_paths(train_images);
	char **paths = (char **)list_to_array(plist);

	int init_w = net.w;
	int init_h = net.h;
	int iter_save, iter_save_last, iter_map;
	iter_save = get_current_batch(net);
	iter_save_last = get_current_batch(net);
	iter_map = get_current_batch(net);
	float mean_average_precision = -1;

	load_args args = { 0 };

	args.w = net.w;
	args.h = net.h;
	args.c = net.c;
	args.paths = paths;
	args.n = imgs;
	args.m = plist->size;
	args.classes = classes;
	args.flip = net.flip;
	args.jitter = jitter;
	args.num_boxes = l.max_boxes;
	args.d = &buffer;
	args.type = DETECTION_DATA;
	args.threads = 16;

	args.angle = net.angle;
	args.exposure = net.exposure;
	args.saturation = net.saturation;
	args.hue = net.hue;

	double time;
	int count = 0;
	const char *base = d->datacfg->name;
	pthread_t load_thread = load_data(args);

	while (get_current_batch(net) < net.max_batches) {
		if (l.random && count++ % 10 == 0) {
			printf("Resizing\n");
			float random_val = rand_scale(1.4);    // *x or /x
			int dim_w = roundl(random_val * init_w / 32) * 32;
			int dim_h = roundl(random_val * init_h / 32) * 32;

			if (dim_w < 32) {
				dim_w = 32;
			}
			if (dim_h < 32) {
				dim_h = 32;
			}
			printf("%d x %d \n", dim_w, dim_h);
			args.w = dim_w;
			args.h = dim_h;

			pthread_join(load_thread, 0);
			train = buffer;
			free_data(train);
			load_thread = load_data(args);

			for (i = 0; i < ngpus; ++i) {
				resize_network(nets + i, dim_w, dim_h);
			}
			net = nets[0];
		}
		time = what_time_is_it_now();
		pthread_join(load_thread, 0);
		train = buffer;
		load_thread = load_data(args);

		printf("Loaded: %lf seconds\n", (what_time_is_it_now() - time));

		time = what_time_is_it_now();
		float loss = 0;
#ifdef GPU
		if (ngpus == 1) {
			loss = train_network(net, train);
		} else {
			loss = train_networks(nets, ngpus, train, 4);
		}
#else
		loss = train_network(net, train);
#endif
		if (avg_loss < 0 || avg_loss != avg_loss) {
			avg_loss = loss;    // if(-inf or nan)
		}
		avg_loss = avg_loss * .9 + loss * .1;

		i = get_current_batch(net);
		if (net.cudnn_half) {
			if (i < net.burn_in) {
				printf("\n Tensor Cores are disabled until the "
				       "first %d iterations are reached.",
				       2 * net.burn_in);
			} else {
				printf("\n Tensor Cores are used.");
			}
		}
		printf(
		    "\n %d: %f, %f avg loss, %f rate, %lf seconds, %d images\n",
		    get_current_batch(net), loss, avg_loss,
		    get_current_rate(net), (what_time_is_it_now() - time),
		    i * imgs);

		if (i >= (iter_save + 1000)) {
			iter_save = i;
#ifdef GPU
			if (ngpus != 1) {
				sync_nets(nets, ngpus, 0);
			}
#endif
			char buff[256];
			sprintf(buff, "%s/%s_%d.weights", backup_directory,
				base, i);
			save_weights(net, buff);
		}
		if (i >= (iter_save_last + 100)) {
			iter_save_last = i;
#ifdef GPU
			if (ngpus != 1) {
				sync_nets(nets, ngpus, 0);
			}
#endif
			char buff[256];
			sprintf(buff, "%s/%s_last.weights", backup_directory,
				base);
			save_weights(net, buff);
		}
		free_data(train);
	}
#ifdef GPU
	if (ngpus != 1) {
		sync_nets(nets, ngpus, 0);
	}
#endif
	char buff[256];
	sprintf(buff, "%s/%s_final.weights", backup_directory, base);
	save_weights(net, buff);

	// free memory
	pthread_join(load_thread, 0);
	free_data(buffer);

	free(paths);
	free_list_contents(plist);
	free_list(plist);

	free_list_contents_kvp(options);
	free_list(options);

	free(nets);
}
