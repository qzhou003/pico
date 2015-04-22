#include "detect-cuda.h"
#include "cascades/face-cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

dim3 compute_grid(dim3 block, int cols, int rows)
{
	int cells_x = cols / block.x;
	if (cols % block.x)
		cells_x += 1;
	int cells_y = rows / block.y;
	if (rows % block.y)
		cells_y += 1;
	return dim3(cells_x, cells_y);
}

int find_faces_cuda(
	float *rs, float *cs, float *ss, float *qs, int maxndetections,
	const uint8_t *pixels, int nrows, int ncols, int ldim,
	float scalefactor, float stridefactor, float minsize, float maxsize)
{
	dim3 block(16, 16);
	dim3 grid = compute_grid(block, ncols, nrows);

	for (float s = minsize; s <= maxsize; s *= scalefactor)
	{
		float dr = std::max(stridefactor * s, 1.0f);
		float dc = dr;

		facedet_cuda<<<grid, block>>>(
			rs, cs, ss, qs, pixels, nrows, ncols, ldim, dr, dc);
	}

}
