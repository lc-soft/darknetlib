#ifndef DARKNET_H
#define DARKNET_H

#if defined(__GNUC__)
#define DARKNET_API
#else
#ifdef DARKNET_EXPORTS
#define DARKNET_API __declspec(dllexport)
#else
#define DARKNET_API
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

DARKNET_BEGIN_HEADER

typedef struct darknet_detector darknet_detector_t;
typedef struct darknet_detection darknet_detection_t;
typedef struct darknet_detections darknet_detections_t;
typedef struct darknet_box darknet_box_t;

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

DARKNET_API void darknet_detections_destroy(darknet_detections_t *d);
DARKNET_API void darknet_detector_destroy(darknet_detector_t *d);

DARKNET_API darknet_detector_t *darnet_detector_create(const char *datacfg,
						       const char *cfgfile,
						       const char *weightfile);

DARKNET_API int darknet_detector_test(darknet_detector_t *d, const char *file,
				      darknet_detections_t *results);

DARKNET_END_HEADER

#endif
