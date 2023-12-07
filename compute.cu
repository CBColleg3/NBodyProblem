#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

__global__ void findDistance(vector3 **accels, vector3 *hPos, double *mass)
{

	int k;

	// first compute the pairwise accelerations.  Effect is on the first argument.s
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= NUMENTITIES || j >= NUMENTITIES)
		return;

	// printf("bruh\n");
	if (i == j)
	{
		FILL_VECTOR(accels[i][j], 0, 0, 0);
	}
	else
	{

		vector3 distance;
		for (k = 0; k < 3; k++)
		{
			distance[k] = hPos[i][k] - hPos[j][k];
		}
		double magnitude_sq = (distance[0] * distance[0]) + (distance[1] * distance[1]) + (distance[2] * distance[2]);
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
		FILL_VECTOR(accels[i][j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
	}
}

__global__ void sumValues(vector3 **accels, vector3 *hPos, vector3 *hVel)
{
	int i = blockIdx.x;
	int k = threadIdx.x;
	int j;

	vector3 accel_sum = {0, 0, 0};
	for (j = 0; j < NUMENTITIES; j++)
	{
		accel_sum[k] += accels[i][j][k];
		//  printf("accels[%d][%d][%d]: %lf\n", i, j, k, accels[i][j][k]);
	}
	// compute the new velocity based on the acceleration and time interval
	// compute the new position based on the velocity and time interval
	hVel[i][k] += accel_sum[k] * INTERVAL;
	hPos[i][k] += hVel[i][k] * INTERVAL;
}

// compute: Updates the positions and locations of the objects in the system based on gravity.
// Parameters: None
// Returns: None
// Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(vector3 **accels, vector3 *hPos, vector3 *hVel, double *mass)
{

	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		printf("something before compute is bad! %s\n", cudaGetErrorString(e));
	}

	dim3 blockSize(16, 16, 3); // 256 threads
	dim3 numBlocks((NUMENTITIES + 15) / blockSize.x, (NUMENTITIES + 15) / blockSize.y);
	// NUMENTITIES = 6 blocks across and 6 down.
	// Rounded up integer division ^. we want more threads then we need.
	// (numBlocks + (blockSize - 1)) / blockSize;

	findDistance<<<numBlocks, blockSize>>>(accels, hPos, mass);
	e = cudaDeviceSynchronize();
	if (e != cudaSuccess)
	{
		printf("find Distance is bad %s\n", cudaGetErrorString(e));
	}

	dim3 grim_dim(NUMENTITIES, 1, 1);
	sumValues<<<grim_dim, 3>>>(accels, hPos, hVel);
	e = cudaDeviceSynchronize();
	if (e != cudaSuccess)
	{
		printf("sum Values is bad %s\n", cudaGetErrorString(e));
	}
}