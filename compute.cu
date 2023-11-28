#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

// compute: Updates the positions and locations of the objects in the system based on gravity.
// Parameters: None
// Returns: None
// Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute()
{
	vector3 *values;
	vector3 **accels;

	cudaMallocManaged(&values, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	cudaMallocManaged(&accels, (sizeof(vector3 *)) * NUMENTITIES);

	findDistance<<<1, 100>>>(values, accels);
	cudaDeviceSynchronize();
	sumValues<<<1, 100>>>(values, accels);

	cudaFree(accels);
	cudaFree(values);
}

__global__ void findDistance(vector3 *values, vector3 **accels)
{
	int i, j, k;
	// worry about size later, one block for each thread lol

	for (i = 0; i < NUMENTITIES; i++)
	{
		accels[i] = &values[i * NUMENTITIES];
	}

	int index = threadIdx.x;
	int stride = blockDim.x; // in the examples case, the stride was 256 (256,0,0)

	// first compute the pairwise accelerations.  Effect is on the first argument.
	for (i = index; i < NUMENTITIES; i += stride)
	{
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

__global__ void sumValues(vector3 *values, vector3 **accels)
{
	int i, j, k;

	// worry about size later, one block for each thread lol

	for (i = 0; i < NUMENTITIES; i++)
	{
		accels[i] = &values[i * NUMENTITIES];
	}

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