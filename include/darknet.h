#ifndef DARKNET_H
#define DARKNET_H

#ifdef DARKNETLIB_API
#undef DARKNETLIB_API
#endif
#if defined(__GNUC__)
#define DARKNETLIB_API extern
#else
#ifdef DARKNETLIB_EXPORTS
#define DARKNETLIB_API __declspec(dllexport)
#else
#define DARKNETLIB_API __declspec(dllimport)
#endif
#endif

#ifdef __cplusplus
#define DARKNET_BEGIN_HEADER extern "C" {
#define DARKNET_END_HEADER }
#else
#define DARKNET_BEGIN_HEADER
#define DARKNET_END_HEADER
#endif

#include <stdlib.h>
#include <setjmp.h>

DARKNET_BEGIN_HEADER

typedef struct darknet_network darknet_network_t;
typedef struct darknet_detector darknet_detector_t;
typedef struct darknet_detection darknet_detection_t;
typedef struct darknet_detections darknet_detections_t;
typedef struct darknet_box darknet_box_t;
typedef struct darknet_config darknet_config_t;
typedef struct darknet_dataconfig darknet_dataconfig_t;

struct darknet_box {
	float x, y, w, h;
};

struct darknet_detection {
	const char *best_name; /**< best name */
	char **names;          /**< matched names */
	float *prob;           /**< probability list */
	size_t names_count;    /**< count of matched names */
	darknet_box_t box;     /**< box */
};

struct darknet_detections {
	darknet_detection_t *list;
	size_t length;
};

typedef enum darknet_error {
	DARKNET_ERROR = 1,
	DARKNET_ARG_ERROR,
	DARKNET_IO_ERROR,
	DARKNET_DETECTOR_ERROR,
	DARKNET_CUDA_ERROR
} darknet_error_t;

DARKNETLIB_API jmp_buf darknet_jmp_buf;
DARKNETLIB_API int darknet_jmp_buf_valid;
DARKNETLIB_API darknet_error_t darknet_last_error;

// clang-format off

#define darknet_try \
	do {\
		darknet_jmp_buf_valid = 1;\
		if (!setjmp(darknet_jmp_buf)) {
#define darknet_catch(ERR) \
		} else {\
			darknet_error_t ERR = darknet_last_error;
#define darknet_etry \
		}\
		darknet_jmp_buf_valid = 0;\
	} while(0)
#define darknet_throw(err, errmsg, ...) \
	darknet_set_error(err, errmsg, ##__VA_ARGS__);\
	if (darknet_jmp_buf_valid) {\
		longjmp(darknet_jmp_buf, err);\
	}

// clang-format on

DARKNETLIB_API void darknet_set_error(darknet_error_t err, const char *format,
				      ...);
DARKNETLIB_API const char *darknet_get_last_error_string(void);

DARKNETLIB_API void darknet_detections_destroy(darknet_detections_t *d);
DARKNETLIB_API void darknet_detector_destroy(darknet_detector_t *d);
DARKNETLIB_API void darknet_network_destroy(darknet_network_t *net);
DARKNETLIB_API void darknet_dataconfig_destroy(darknet_dataconfig_t *cfg);
DARKNETLIB_API void darknet_config_destroy(darknet_config_t *cfg);

DARKNETLIB_API darknet_config_t *darknet_config_load(const char *file);

DARKNETLIB_API size_t darknet_config_set_workdir(darknet_config_t *cfg,
						 const char *workdir);

DARKNETLIB_API darknet_dataconfig_t *darknet_dataconfig_load(const char *file);

DARKNETLIB_API int darknet_dataconfig_get_classes(darknet_dataconfig_t *cfg);

DARKNETLIB_API size_t darknet_dataconfig_set_workdir(darknet_dataconfig_t *cfg,
						     const char *workdir);

DARKNETLIB_API darknet_network_t *darknet_network_create(darknet_config_t *cfg);

DARKNETLIB_API int darknet_network_load_weights(darknet_network_t *net,
						const char *weightfile);

DARKNETLIB_API darknet_detector_t *darknet_detector_create(
    darknet_network_t *net, darknet_dataconfig_t *cfg);

DARKNETLIB_API int darknet_detector_test(darknet_detector_t *d,
					 const char *file,
					 darknet_detections_t *results);

DARKNETLIB_API void darknet_detector_train(darknet_detector_t *d);

DARKNET_END_HEADER

#endif
