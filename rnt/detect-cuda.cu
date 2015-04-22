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

	thrust::device_vector<unsigned char> pixels_d(nrows * ncols);
	thrust::copy_n(pixels, nrows * ncols, pixels_d.begin());
	std::vector<unsigned char> result(ncols * nrows);
	std::vector<unsigned char> response(ncols * nrows);
	int ndetections = 0;
	for (float s = minsize; s <= maxsize; s *= scalefactor)
	{
		float dr = std::max(stridefactor * s, 1.0f);
		float dc = dr;

		int res_cols = int((ncols - s) / dc) + 1;
		int res_rows = int((nrows - s) / dr) + 1;
		thrust::device_vector<unsigned char> result_d(res_cols * res_rows);
		thrust::device_vector<float> response_d(res_cols * res_rows);

		dim3 grid = compute_grid(block, ncols, nrows);
		facedet_cuda<<<grid, block>>>(
			thrust::raw_pointer_cast(&response_d[0]),
			thrust::raw_pointer_cast(&result_d[0]),
			int(s), thrust::raw_pointer_cast(&pixels_d[0]),
			nrows, ncols, ncols, dr, dc, res_cols);

		thrust::copy(result_d.begin(), result_d.end(), result.begin());
		thrust::copy(response_d.begin(), response_d.end(), response.begin());
		float row = s / 2 + 1;
		for (int r = 0; row <= nrows-s/2-1; row += dr, r += 1)
		{
			float col = s / 2 + 1;
			for (int c = 0; col <= ncols-s/2-1; col += dc, c += 1)
			{
				if (ndetections >= maxndetections)
					break;
				if (!result[r * res_cols + c])
					continue;

				qs[ndetections] = response[r * res_cols + c];
				rs[ndetections] = row;
				cs[ndetections] = col;
				ss[ndetections] = s;
				++ndetections;
			}
		}
	}
	return ndetections;
}
