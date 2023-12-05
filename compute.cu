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

	// int index = threadIdx.x;
	// int stride = blockDim.x;

	// if i or j >= NUMENTITIES then return

	// first compute the pairwise accelerations.  Effect is on the first argument.
	for (i = 0; i < NUMENTITIES; i++)
	{
		// printf("bruh\n");
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

	// Similar thing for i and j here I believe

	// int index = threadIdx.x;
	// int stride = blockDim.x; // in the examples case, the stride was 256 (256,0,0)

	for (i = 0; i < NUMENTITIES; i++)
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

		// Silber did an additional kernel here to complete his reduction, u have
		// to do syncthreads if ur doing another kernel. cuts about a second of runtime.
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
void compute(vector3 **accels, vector3 *hPos, vector3 *hVel, double *mass)
{

	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		printf("something before compute is bad! %s\n", cudaGetErrorString(e));
	}
	// printf("printing device variables\n");
	// printDeviceSystem(stdout);

	// My kernel is not firing;

	findDistance<<<1, 1>>>(accels, hPos, hVel, mass);
	e = cudaDeviceSynchronize();
	if (e != cudaSuccess)
	{
		printf("find Distance is bad %s\n", cudaGetErrorString(e));
	}

	sumValues<<<1, 1>>>(accels, hPos, hVel);
	e = cudaDeviceSynchronize();
	if (e != cudaSuccess)
	{
		printf("sum Values is bad %s\n", cudaGetErrorString(e));
	}
}