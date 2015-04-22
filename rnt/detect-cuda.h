#ifndef DETECTCUDA_H
#define DETECTCUDA_H

#include <stdint.h>

int find_faces_cuda(
	float *rs, float *cs, float *ss, float *qs, int maxndetections,
	const uint8_t *pixels, int nrows, int ncols, int ldim,
	float scalefactor, float stridefactor, float minsize, float maxsize);

#endif // DETECTCUDA_H
