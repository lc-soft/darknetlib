#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <limits.h>
#ifdef WIN32
#include "unistd.h"
#include "gettimeofday.h"
#else
#include <unistd.h>
#include <sys/time.h>
#endif

#include "darknet.h"
#include "utils.h"

#pragma warning(disable : 4996)

double what_time_is_it_now()
{
	struct timeval time;
	if (gettimeofday(&time, NULL)) {
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int *read_map(char *filename)
{
	int n = 0;
	int *map = 0;
	char *str;
	FILE *file = fopen(filename, "r");
	if (!file)
		file_error(filename);
	while ((str = fgetl(file))) {
		++n;
		map = realloc(map, n * sizeof(int));
		map[n - 1] = atoi(str);
	}
	return map;
}

void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections)
{
	size_t i;
	for (i = 0; i < sections; ++i) {
		size_t start = n * i / sections;
		size_t end = n * (i + 1) / sections;
		size_t num = end - start;
		shuffle((char *)arr + (start * size), num, size);
	}
}

void shuffle(void *arr, size_t n, size_t size)
{
	size_t i;
	void *swp = calloc(1, size);
	for (i = 0; i < n - 1; ++i) {
		size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
		memcpy(swp, (char *)arr + (j * size), size);
		memcpy((char *)arr + (j * size), (char *)arr + (i * size),
		       size);
		memcpy((char *)arr + (i * size), swp, size);
	}
}

void del_arg(int argc, char **argv, int index)
{
	int i;
	for (i = index; i < argc - 1; ++i) argv[i] = argv[i + 1];
	argv[i] = 0;
}

int find_arg(int argc, char *argv[], char *arg)
{
	int i;
	for (i = 0; i < argc; ++i) {
		if (!argv[i])
			continue;
		if (0 == strcmp(argv[i], arg)) {
			del_arg(argc, argv, i);
			return 1;
		}
	}
	return 0;
}

int find_int_arg(int argc, char **argv, char *arg, int def)
{
	int i;
	for (i = 0; i < argc - 1; ++i) {
		if (!argv[i])
			continue;
		if (0 == strcmp(argv[i], arg)) {
			def = atoi(argv[i + 1]);
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}

float find_float_arg(int argc, char **argv, char *arg, float def)
{
	int i;
	for (i = 0; i < argc - 1; ++i) {
		if (!argv[i])
			continue;
		if (0 == strcmp(argv[i], arg)) {
			def = atof(argv[i + 1]);
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}

char *find_char_arg(int argc, char **argv, char *arg, char *def)
{
	int i;
	for (i = 0; i < argc - 1; ++i) {
		if (!argv[i])
			continue;
		if (0 == strcmp(argv[i], arg)) {
			def = argv[i + 1];
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}

char *basecfg(char *cfgfile)
{
	char *c = cfgfile;
	char *next;
	while ((next = strchr(c, '/'))) {
		c = next + 1;
	}
	if (!next)
		while ((next = strchr(c, '\\'))) {
			c = next + 1;
		}
	c = copy_string(c);
	next = strchr(c, '.');
	if (next)
		*next = 0;
	return c;
}

int alphanum_to_int(char c)
{
	return (c < 58) ? c - 48 : c - 87;
}
char int_to_alphanum(int i)
{
	if (i == 36)
		return '.';
	return (i < 10) ? i + 48 : i + 87;
}

void pm(int M, int N, float *A)
{
	int i, j;
	for (i = 0; i < M; ++i) {
		printf("%d ", i + 1);
		for (j = 0; j < N; ++j) {
			printf("%2.4f, ", A[i * N + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void find_replace(char *str, char *orig, char *rep, char *output)
{
	char *buffer = calloc(8192, sizeof(char));
	char *p;

	sprintf(buffer, "%s", str);
	if (!(p = strstr(buffer, orig))) {    // Is 'orig' even in 'str'?
		sprintf(output, "%s", str);
		free(buffer);
		return;
	}

	*p = '\0';

	sprintf(output, "%s%s%s", buffer, rep, p + strlen(orig));
	free(buffer);
}

void trim(char *str)
{
	char *buffer = calloc(8192, sizeof(char));
	sprintf(buffer, "%s", str);

	char *p = buffer;
	while (*p == ' ' || *p == '\t') ++p;

	char *end = p + strlen(p) - 1;
	while (*end == ' ' || *end == '\t') {
		*end = '\0';
		--end;
	}
	sprintf(str, "%s", p);

	free(buffer);
}

void find_replace_extension(char *str, char *orig, char *rep, char *output)
{
	char *buffer = calloc(8192, sizeof(char));

	sprintf(buffer, "%s", str);
	char *p = strstr(buffer, orig);
	int offset = (p - buffer);
	int chars_from_end = strlen(buffer) - offset;
	if (!p || chars_from_end !=
		      strlen(orig)) {    // Is 'orig' even in 'str' AND is
					 // 'orig' found at the end of 'str'?
		sprintf(output, "%s", str);
		free(buffer);
		return;
	}

	*p = '\0';
	sprintf(output, "%s%s%s", buffer, rep, p + strlen(orig));
	free(buffer);
}

void replace_image_to_label(char *input_path, char *output_path)
{
	find_replace(input_path, "/images/train2014/", "/labels/train2014/",
		     output_path);    // COCO
	find_replace(output_path, "/images/val2014/", "/labels/val2014/",
		     output_path);    // COCO
	find_replace(output_path, "/JPEGImages/", "/labels/",
		     output_path);    // PascalVOC
	find_replace(output_path, "\\images\\train2014\\",
		     "\\labels\\train2014\\", output_path);    // COCO
	find_replace(output_path, "\\images\\val2014\\", "\\labels\\val2014\\",
		     output_path);    // COCO
	find_replace(output_path, "\\JPEGImages\\", "\\labels\\",
		     output_path);    // PascalVOC
	// find_replace(output_path, "/images/", "/labels/", output_path);    //
	// COCO find_replace(output_path, "/VOC2007/JPEGImages/",
	// "/VOC2007/labels/", output_path);        // PascalVOC
	// find_replace(output_path, "/VOC2012/JPEGImages/", "/VOC2012/labels/",
	// output_path);        // PascalVOC

	// find_replace(output_path, "/raw/", "/labels/", output_path);
	trim(output_path);

	// replace only ext of files
	find_replace_extension(output_path, ".jpg", ".txt", output_path);
	find_replace_extension(output_path, ".JPG", ".txt",
			       output_path);    // error
	find_replace_extension(output_path, ".jpeg", ".txt", output_path);
	find_replace_extension(output_path, ".JPEG", ".txt", output_path);
	find_replace_extension(output_path, ".png", ".txt", output_path);
	find_replace_extension(output_path, ".PNG", ".txt", output_path);
	find_replace_extension(output_path, ".bmp", ".txt", output_path);
	find_replace_extension(output_path, ".BMP", ".txt", output_path);
	find_replace_extension(output_path, ".ppm", ".txt", output_path);
	find_replace_extension(output_path, ".PPM", ".txt", output_path);
}

float sec(clock_t clocks)
{
	return (float)clocks / CLOCKS_PER_SEC;
}

void top_k(float *a, int n, int k, int *index)
{
	int i, j;
	for (j = 0; j < k; ++j) index[j] = -1;
	for (i = 0; i < n; ++i) {
		int curr = i;
		for (j = 0; j < k; ++j) {
			if ((index[j] < 0) || a[curr] > a[index[j]]) {
				int swap = curr;
				curr = index[j];
				index[j] = swap;
			}
		}
	}
}

void error(const char *s)
{
	perror(s);
	assert(0);
	exit(EXIT_FAILURE);
}

void malloc_error()
{
	fprintf(stderr, "Malloc error\n");
	exit(EXIT_FAILURE);
}

void file_error(char *s)
{
	darknet_throw(DARKNET_IO_ERROR, "couldn't open file: %s", s);
	exit(EXIT_FAILURE);
}

list *split_str(char *s, char delim)
{
	size_t i;
	size_t len = strlen(s);
	list *l = make_list();
	list_insert(l, s);
	for (i = 0; i < len; ++i) {
		if (s[i] == delim) {
			s[i] = '\0';
			list_insert(l, &(s[i + 1]));
		}
	}
	return l;
}

void strip(char *s)
{
	size_t i;
	size_t len = strlen(s);
	size_t offset = 0;
	for (i = 0; i < len; ++i) {
		char c = s[i];
		if (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
		    c == 0x0d || c == 0x0a)
			++offset;
		else
			s[i - offset] = c;
	}
	s[len - offset] = '\0';
}

void strip_args(char *s)
{
	size_t i;
	size_t len = strlen(s);
	size_t offset = 0;
	for (i = 0; i < len; ++i) {
		char c = s[i];
		if (c == '\t' || c == '\n' || c == '\r' || c == 0x0d ||
		    c == 0x0a)
			++offset;
		else
			s[i - offset] = c;
	}
	s[len - offset] = '\0';
}

void strip_char(char *s, char bad)
{
	size_t i;
	size_t len = strlen(s);
	size_t offset = 0;
	for (i = 0; i < len; ++i) {
		char c = s[i];
		if (c == bad)
			++offset;
		else
			s[i - offset] = c;
	}
	s[len - offset] = '\0';
}

void free_ptrs(void **ptrs, int n)
{
	int i;
	for (i = 0; i < n; ++i) free(ptrs[i]);
	free(ptrs);
}

char *fgetl(FILE *fp)
{
	if (feof(fp))
		return 0;
	size_t size = 512;
	char *line = malloc(size * sizeof(char));
	if (!fgets(line, size, fp)) {
		free(line);
		return 0;
	}

	size_t curr = strlen(line);

	while ((line[curr - 1] != '\n') && !feof(fp)) {
		if (curr == size - 1) {
			size *= 2;
			line = realloc(line, size * sizeof(char));
			if (!line) {
				printf("%ld\n", size);
				malloc_error();
			}
		}
		size_t readsize = size - curr;
		if (readsize > INT_MAX)
			readsize = INT_MAX - 1;
		fgets(&line[curr], readsize, fp);
		curr = strlen(line);
	}
	if (curr >= 2)
		if (line[curr - 2] == 0x0d)
			line[curr - 2] = 0x00;

	if (curr >= 1)
		if (line[curr - 1] == 0x0a)
			line[curr - 1] = 0x00;

	return line;
}

int read_int(int fd)
{
	int n = 0;
	int next = read(fd, &n, sizeof(int));
	if (next <= 0)
		return -1;
	return n;
}

void write_int(int fd, int n)
{
	int next = write(fd, &n, sizeof(int));
	if (next <= 0)
		error("read failed");
}

int read_all_fail(int fd, char *buffer, size_t bytes)
{
	size_t n = 0;
	while (n < bytes) {
		int next = read(fd, buffer + n, bytes - n);
		if (next <= 0)
			return 1;
		n += next;
	}
	return 0;
}

int write_all_fail(int fd, char *buffer, size_t bytes)
{
	size_t n = 0;
	while (n < bytes) {
		size_t next = write(fd, buffer + n, bytes - n);
		if (next <= 0)
			return 1;
		n += next;
	}
	return 0;
}

void read_all(int fd, char *buffer, size_t bytes)
{
	size_t n = 0;
	while (n < bytes) {
		int next = read(fd, buffer + n, bytes - n);
		if (next <= 0)
			error("read failed");
		n += next;
	}
}

void write_all(int fd, char *buffer, size_t bytes)
{
	size_t n = 0;
	while (n < bytes) {
		size_t next = write(fd, buffer + n, bytes - n);
		if (next <= 0)
			error("write failed");
		n += next;
	}
}

char *copy_string(char *s)
{
	char *copy = malloc(strlen(s) + 1);
	strncpy(copy, s, strlen(s) + 1);
	return copy;
}

list *parse_csv_line(char *line)
{
	list *l = make_list();
	char *c, *p;
	int in = 0;
	for (c = line, p = line; *c != '\0'; ++c) {
		if (*c == '"')
			in = !in;
		else if (*c == ',' && !in) {
			*c = '\0';
			list_insert(l, copy_string(p));
			p = c + 1;
		}
	}
	list_insert(l, copy_string(p));
	return l;
}

int count_fields(char *line)
{
	int count = 0;
	int done = 0;
	char *c;
	for (c = line; !done; ++c) {
		done = (*c == '\0');
		if (*c == ',' || done)
			++count;
	}
	return count;
}

float *parse_fields(char *line, int n)
{
	float *field = calloc(n, sizeof(float));
	char *c, *p, *end;
	int count = 0;
	int done = 0;
	for (c = line, p = line; !done; ++c) {
		done = (*c == '\0');
		if (*c == ',' || done) {
			*c = '\0';
			field[count] = strtod(p, &end);
			if (p == c)
				field[count] = nan("");
			if (end != c && (end != c - 1 || *end != '\r'))
				field[count] = nan("");    // DOS file formats!
			p = c + 1;
			++count;
		}
	}
	return field;
}

float sum_array(float *a, int n)
{
	int i;
	float sum = 0;
	for (i = 0; i < n; ++i) sum += a[i];
	return sum;
}

float mean_array(float *a, int n)
{
	return sum_array(a, n) / n;
}

void mean_arrays(float **a, int n, int els, float *avg)
{
	int i;
	int j;
	memset(avg, 0, els * sizeof(float));
	for (j = 0; j < n; ++j) {
		for (i = 0; i < els; ++i) {
			avg[i] += a[j][i];
		}
	}
	for (i = 0; i < els; ++i) {
		avg[i] /= n;
	}
}

void print_statistics(float *a, int n)
{
	float m = mean_array(a, n);
	float v = variance_array(a, n);
	printf("MSE: %.6f, Mean: %.6f, Variance: %.6f\n", mse_array(a, n), m,
	       v);
}

float variance_array(float *a, int n)
{
	int i;
	float sum = 0;
	float mean = mean_array(a, n);
	for (i = 0; i < n; ++i) sum += (a[i] - mean) * (a[i] - mean);
	float variance = sum / n;
	return variance;
}

int constrain_int(int a, int min, int max)
{
	if (a < min)
		return min;
	if (a > max)
		return max;
	return a;
}

float constrain(float min, float max, float a)
{
	if (a < min)
		return min;
	if (a > max)
		return max;
	return a;
}

float dist_array(float *a, float *b, int n, int sub)
{
	int i;
	float sum = 0;
	for (i = 0; i < n; i += sub) sum += pow(a[i] - b[i], 2);
	return sqrt(sum);
}

float mse_array(float *a, int n)
{
	int i;
	float sum = 0;
	for (i = 0; i < n; ++i) sum += a[i] * a[i];
	return sqrt(sum / n);
}

void normalize_array(float *a, int n)
{
	int i;
	float mu = mean_array(a, n);
	float sigma = sqrt(variance_array(a, n));
	for (i = 0; i < n; ++i) {
		a[i] = (a[i] - mu) / sigma;
	}
	mu = mean_array(a, n);
	sigma = sqrt(variance_array(a, n));
}

void translate_array(float *a, int n, float s)
{
	int i;
	for (i = 0; i < n; ++i) {
		a[i] += s;
	}
}

float mag_array(float *a, int n)
{
	int i;
	float sum = 0;
	for (i = 0; i < n; ++i) {
		sum += a[i] * a[i];
	}
	return sqrt(sum);
}

void scale_array(float *a, int n, float s)
{
	int i;
	for (i = 0; i < n; ++i) {
		a[i] *= s;
	}
}

int sample_array(float *a, int n)
{
	float sum = sum_array(a, n);
	scale_array(a, n, 1. / sum);
	float r = rand_uniform(0, 1);
	int i;
	for (i = 0; i < n; ++i) {
		r = r - a[i];
		if (r <= 0)
			return i;
	}
	return n - 1;
}

int max_index(float *a, int n)
{
	if (n <= 0)
		return -1;
	int i, max_i = 0;
	float max = a[0];
	for (i = 1; i < n; ++i) {
		if (a[i] > max) {
			max = a[i];
			max_i = i;
		}
	}
	return max_i;
}

int int_index(int *a, int val, int n)
{
	int i;
	for (i = 0; i < n; ++i) {
		if (a[i] == val)
			return i;
	}
	return -1;
}

int rand_int(int min, int max)
{
	if (max < min) {
		int s = min;
		min = max;
		max = s;
	}
	int r = (rand() % (max - min + 1)) + min;
	return r;
}

// From http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
float rand_normal()
{
	static int haveSpare = 0;
	static double rand1, rand2;

	if (haveSpare) {
		haveSpare = 0;
		return sqrt(rand1) * sin(rand2);
	}

	haveSpare = 1;

	rand1 = rand() / ((double)RAND_MAX);
	if (rand1 < 1e-100)
		rand1 = 1e-100;
	rand1 = -2 * log(rand1);
	rand2 = (rand() / ((double)RAND_MAX)) * TWO_PI;

	return sqrt(rand1) * cos(rand2);
}

/*
   float rand_normal()
   {
   int n = 12;
   int i;
   float sum= 0;
   for(i = 0; i < n; ++i) sum += (float)rand()/RAND_MAX;
   return sum-n/2.;
   }
 */

size_t rand_size_t()
{
	return ((size_t)(rand() & 0xff) << 56) |
	       ((size_t)(rand() & 0xff) << 48) |
	       ((size_t)(rand() & 0xff) << 40) |
	       ((size_t)(rand() & 0xff) << 32) |
	       ((size_t)(rand() & 0xff) << 24) |
	       ((size_t)(rand() & 0xff) << 16) |
	       ((size_t)(rand() & 0xff) << 8) | ((size_t)(rand() & 0xff) << 0);
}

float rand_uniform(float min, float max)
{
	if (max < min) {
		float swap = min;
		min = max;
		max = swap;
	}
	return ((float)rand() / RAND_MAX * (max - min)) + min;
	// return (random_float() * (max - min)) + min;
}

float rand_scale(float s)
{
	float scale = rand_uniform_strong(1, s);
	if (random_gen() % 2)
		return scale;
	return 1. / scale;
}

float **one_hot_encode(float *a, int n, int k)
{
	int i;
	float **t = calloc(n, sizeof(float *));
	for (i = 0; i < n; ++i) {
		t[i] = calloc(k, sizeof(float));
		int index = (int)a[i];
		t[i][index] = 1;
	}
	return t;
}

unsigned int random_gen()
{
	unsigned int rnd = 0;
#ifdef WIN32
	rand_s(&rnd);
#else
	rnd = rand();
#endif
	return rnd;
}

float random_float()
{
#ifdef WIN32
	return ((float)random_gen() / (float)UINT_MAX);
#else
	return ((float)random_gen() / (float)RAND_MAX);
#endif
}

float rand_uniform_strong(float min, float max)
{
	if (max < min) {
		float swap = min;
		min = max;
		max = swap;
	}
	return (random_float() * (max - min)) + min;
}
