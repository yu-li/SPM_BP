#include "FlowInputOutput.h"

void ReadFlowFile(float *motion, const char* filename, int height, int width)
{
	FILE *stream = fopen(filename, "rb");
	if (stream == 0)
		printf("ReadFlowFile: could not open %s", filename);

	float tag;

	if ((int)fread(&tag,    sizeof(float), 1, stream) != 1 ||
		(int)fread(&width,  sizeof(int),   1, stream) != 1 ||
		(int)fread(&height, sizeof(int),   1, stream) != 1)
		printf("ReadFlowFile: problem reading file %s", filename);

	if (tag != TAG_FLOAT) // simple test for correct endian-ness
		printf("ReadFlowFile(%s): wrong tag (possibly due to big-endian machine?)", filename);

	// another sanity check to see that integers were read correctly (99999 should do the trick...)
	if (width < 1 || width > 99999)
		printf("ReadFlowFile(%s): illegal width %d", filename, width);

	if (height < 1 || height > 99999)
		printf("ReadFlowFile(%s): illegal height %d", filename, height);

	int nBands = 2;

	//printf("reading %d x %d x 2 = %d floats\n", width, height, width*height*2);
	int n = nBands * width;
	for (int y = 0; y < height; y++) {
		float* ptr = &motion[y*2*width];
		if ((int)fread(ptr, sizeof(float), n, stream) != n)
			printf("ReadFlowFile(%s): file is too short", filename);
	}

	if (fgetc(stream) != EOF)
		printf("ReadFlowFile(%s): file is too long", filename);

	fclose(stream);
}

void WriteFlowFile(float *motion, const char* filename, int height, int width)
{
	FILE *stream = fopen(filename, "wb");
	if (stream == 0)
		printf("WriteFlowFile: could not open %s", filename);

	// write the header
	fprintf(stream, TAG_STRING);
	if ((int)fwrite(&width,  sizeof(int),   1, stream) != 1 ||
		(int)fwrite(&height, sizeof(int),   1, stream) != 1)
		printf("WriteFlowFile(%s): problem writing header", filename);


	// write the rows
	int n = 2*width;
	for (int y = 0; y < height; y++) {
		float* ptr = &motion[y*2*width];
		if ((int)fwrite(ptr, sizeof(float), n, stream) != n)
			printf("WriteFlowFile(%s): problem writing data", filename);
	}

	fclose(stream);
}

