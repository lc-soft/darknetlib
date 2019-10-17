#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include "../include/darknet.h"

jmp_buf darknet_jmp_buf;
int darknet_jmp_buf_valid = 0;
darknet_error_t darknet_last_error = 0;

static char error_string[1024] = { 0 };

void darknet_set_error(darknet_error_t err, const char *format, ...)
{
	va_list args;

	darknet_last_error = err;
	if (format) {
		va_start(args, format);
		vsnprintf(error_string, 1023, format, args);
		va_end(args);
	} else {
		sprintf(error_string, "no error message");
	}
	if (!darknet_jmp_buf_valid) {
		printf("error: %s\n", error_string);
	}
}

const char *darknet_get_last_error_string(void)
{
	return error_string;
}
