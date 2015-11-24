#ifndef __FLOW_INPUT_OUTPUT
#define __FLOW_INPUT_OUTPUT

#include <cstdio>

#define TAG_FLOAT 202021.25  // check for this when READING the file
#define TAG_STRING "PIEH"    // use this when WRITING the file

void ReadFlowFile(float *motion, const char* filename, int height, int width);
void WriteFlowFile(float *motion, const char* filename, int height, int width);

#endif