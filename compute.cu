#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

__global__ void findDistance(vector3 **accels, vector3 *hPos, vector3 *hVel, double *mass)
{
	// printf("findDIstance is here\n");
	int i, j, k;
	// worry about size later, one block for each thread lol

	int index = threadIdx.x;
	int stride = blockDim.x; // in the examples case, the stride was 256 (256,0,0)

	// first compute the pairwise accelerations.  Effect is on the first argument.
	for (i = index; i < NUMENTITIES; i += stride)
	{
		printf("iterations:%d ,", i);
		for (j = 0; j < NUMENTITIES; j++)
		{
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
	}
}

__global__ void sumValues(vector3 **accels, vector3 *hPos, vector3 *hVel)
{
	int i, j, k;

	// worry about size later, one block for each thread lol

	int index = threadIdx.x;
	int stride = blockDim.x; // in the examples case, the stride was 256 (256,0,0)

	for (i = index; i < NUMENTITIES; i += stride)
	{
		vector3 accel_sum = {0, 0, 0};
		for (j = 0; j < NUMENTITIES; j++)
		{
			for (k = 0; k < 3; k++)
			{
				accel_sum[k] += accels[i][j][k];
			}
		}
		// compute the new velocity based on the acceleration and time interval
		// compute the new position based on the velocity and time interval
		for (k = 0; k < 3; k++)
		{
			hVel[i][k] += accel_sum[k] * INTERVAL;
			hPos[i][k] += hVel[i][k] * INTERVAL;
		}
	}
}

// compute: Updates the positions and locations of the objects in the system based on gravity.
// Parameters: None
// Returns: None
// Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(vector3 *hPos, vector3 *hVel, double *mass)
{
	// printf("Entered into compute\n");
	// fflush(stdout);

	vector3 *values;
	vector3 **accels;

	// printf("Hello3\n");
	// fflush(stdout);

	values = (vector3 *)malloc(sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	vector3 **tempAccel = (vector3 **)malloc(sizeof(vector3 *) * NUMENTITIES);

	// printf("Hello4\n");
	// fflush(stdout);

	for (int i = 0; i < NUMENTITIES; i++)
	{
		tempAccel[i] = &values[i * NUMENTITIES];
	} // make temp arr and do memcpy for non malloc managed version of accels.

	cudaMalloc(&accels, (sizeof(vector3 *)) * NUMENTITIES);
	cudaMemcpy(accels, tempAccel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);

	// printf("%lf\n", hPos[0][1]);
	// fflush(stdout);
	findDistance<<<1, 256>>>(accels, hPos, hVel, mass);
	// cudaDeviceSynchronize();

	// printf("full ran findDistance\n");
	// fflush(stdout);

	sumValues<<<1, 256>>>(accels, hPos, hVel);
	// cudaDeviceSynchronize();
	cudaFree(accels);
	cudaFree(values);
}