#ifndef DARKNET_BUILD_H
#define DARKNET_BUILD_H

#ifdef _WIN32
#define WIN32 1
#endif

#include <stdlib.h>

#define getenv

#define DEBUG_MSG(format, ...)                                           \
	printf("%s:%d: %s(): " format, __FILE__, __LINE__, __FUNCTION__, \
	       ##__VA_ARGS__)

#endif
