#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "planets.h"
#include "compute.h"

// represents the objects in the system.  Global variables
// GLOBAL VARIABLES
// device = GPU variables, d_hVel, and d_hPos are there.
vector3 *hVel, *d_hVel;
vector3 *hPos, *d_hPos;
double *mass, *d_mass;

vector3 *values;
vector3 **accels;

// initHostMemory: Create storage for numObjects entities in our system
// Parameters: numObjects: number of objects to allocate
// Returns: None
// Side Effects: Allocates memory in the hVel, hPos, and mass global variables
void initHostMemory(int numObjects)
{
	hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
	hPos = (vector3 *)malloc(sizeof(vector3) * numObjects);
	mass = (double *)malloc(sizeof(double) * numObjects);

	if (hVel == NULL)
	{
		printf("hVel malloc failed!\n");
	}
	if (hPos == NULL)
	{
		printf("hPos malloc failed!");
	}
	if (mass == NULL)
	{
		printf("mass malloc failed!");
	}
}

void initDeviceMemory(int numObjects)
{
	cudaError_t dVelMalloc = cudaMalloc(&d_hVel, sizeof(vector3) * numObjects);
	cudaError_t dPosMalloc = cudaMalloc(&d_hPos, sizeof(vector3) * numObjects);
	cudaError_t dmassMalloc = cudaMalloc(&d_mass, sizeof(double) * numObjects);

	if (dVelMalloc != cudaSuccess)
	{
		printf("dVelMalloc is bad\n");
	}
	if (dPosMalloc != cudaSuccess)
	{
		printf("dPosMalloc is bad\n");
	}
	if (dmassMalloc != cudaSuccess)
	{
		printf("dMassMalloc is bad\n");
	}
}

void loadDeviceMemory(int numObjects)
{
	cudaError_t hVelCpy = cudaMemcpy(d_hVel, hVel, sizeof(vector3) * numObjects, cudaMemcpyHostToDevice);
	cudaError_t hPosCpy = cudaMemcpy(d_hPos, hPos, sizeof(vector3) * numObjects, cudaMemcpyHostToDevice);
	cudaError_t dMassCpy = cudaMemcpy(d_mass, mass, sizeof(double) * numObjects, cudaMemcpyHostToDevice);

	if (hVelCpy != cudaSuccess)
	{
		printf("hVelCpy is bad %s\n", cudaGetErrorString(hVelCpy));
	}
	if (hPosCpy != cudaSuccess)
	{
		printf("hPosCpy is bad %s\n", cudaGetErrorString(hPosCpy));
	}
	if (dMassCpy != cudaSuccess)
	{
		printf("dMassCpy is bad %s\n", cudaGetErrorString(dMassCpy));
	}
}

// freeHostMemory: Free storage allocated by a previous call to initHostMemory
// Parameters: None
// Returns: None
// Side Effects: Frees the memory allocated to global variables hVel, hPos, and mass.
void freeHostMemory()
{
	free(hVel);
	free(hPos);
	free(mass);
}

void freeDeviceMemory()
{
	cudaFree(hVel);
	cudaFree(hPos);
	cudaFree(mass);
}

// planetFill: Fill the first NUMPLANETS+1 entries of the entity arrays with an estimation
//				of our solar system (Sun+NUMPLANETS)
// Parameters: None
// Returns: None
// Fills the first 8 entries of our system with an estimation of the sun plus our 8 planets.
void planetFill()
{
	int i, j;
	double data[][7] = {SUN, MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE};
	for (i = 0; i <= NUMPLANETS; i++)
	{
		for (j = 0; j < 3; j++)
		{
			hPos[i][j] = data[i][j];
			hVel[i][j] = data[i][j + 3];
		}
		mass[i] = data[i][6];
	}
}

// randomFill: FIll the rest of the objects in the system randomly starting at some entry in the list
// Parameters: 	start: The index of the first open entry in our system (after planetFill).
//				count: The number of random objects to put into our system
// Returns: None
// Side Effects: Fills count entries in our system starting at index start (0 based)
void randomFill(int start, int count)
{
	int i, j, c = start;
	for (i = start; i < start + count; i++)
	{
		for (j = 0; j < 3; j++)
		{
			hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
			hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
			mass[i] = (double)rand() / RAND_MAX * MAX_MASS;
		}
	}
}

// printSystem: Prints out the entire system to the supplied file
// Parameters: 	handle: A handle to an open file with write access to prnt the data to
// Returns: 		none
// Side Effects: Modifies the file handle by writing to it.
void printSystem(FILE *handle)
{
	int i, j;
	for (i = 0; i < NUMENTITIES; i++)
	{
		fprintf(handle, "pos=(");
		for (j = 0; j < 3; j++)
		{
			fprintf(handle, "%lf,", hPos[i][j]);
		}
		printf("),v=(");
		for (j = 0; j < 3; j++)
		{
			fprintf(handle, "%lf,", hVel[i][j]);
		}
		fprintf(handle, "),m=%lf\n", mass[i]);
	}
}

void initAccels()
{
	// The cudaMallocs and frees should be in nbody because u compute these everytime for parallel version, definitely a speed issue.
	//  next should be cudamalloc
	cudaMalloc(&values, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		printf("values Malloc is bad! %s\n", cudaGetErrorString(e));
	}

	cudaMalloc(&accels, (sizeof(vector3 *)) * NUMENTITIES);
	e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		printf("cudaMalloc accels is bad! %s\n", cudaGetErrorString(e));
	}

	vector3 **tempAccel = (vector3 **)malloc(sizeof(vector3 *) * NUMENTITIES);
	if (tempAccel == NULL)
	{
		printf("tempAccel malloc is bad!\n");
	}

	for (int i = 0; i < NUMENTITIES; i++)
	{
		tempAccel[i] = &values[i * NUMENTITIES];
		if (tempAccel[i] == NULL)
		{
			printf("This TempAccel[i] doesn't exist!!\n");
		}
	}

	// MemCopy Accels is bad.
	cudaMemcpy(accels, tempAccel, sizeof(vector3 *) * NUMENTITIES, cudaMemcpyHostToDevice);
	e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		printf("memCopy accels is bad! %s\n", cudaGetErrorString(e));
	}
	free(tempAccel);
}

void copyDeviceToHost()
{
	// cudamemcopy hPos and hVel. mass doesn't change so we dont care about mass.
	cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);

	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		printf("cudaMemcpy hVel dhVel is bad! %s\n", cudaGetErrorString(e));
	}

	cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		printf("cudaMemcpy hPos dhPos is bad! %s\n", cudaGetErrorString(e));
	}
}

int main(int argc, char **argv)
{
	clock_t t0 = clock();
	int t_now;
	// srand(time(NULL));
	srand(1234);

	initHostMemory(NUMENTITIES);
	initDeviceMemory(NUMENTITIES);
	planetFill();
	randomFill(NUMPLANETS + 1, NUMASTEROIDS);

	loadDeviceMemory(NUMENTITIES);
	// now we have a system.
#ifdef DEBUG
	printf("printing system for the first time.\n");
	printSystem(stdout);
#endif

	initAccels();
	for (t_now = 0; t_now < DURATION; t_now += INTERVAL)
	{
		// printf("d_hPos %lf, d_hVel %lf, d_mass %lf", *d_hPos, *d_hVel, *d_mass);
		compute(accels, d_hPos, d_hVel, d_mass);
	}
	copyDeviceToHost();
	clock_t t1 = clock() - t0;
#ifdef DEBUG
	printf("printing system after compute.\n");
	printSystem(stdout);
#endif
	printf("This took a total time of %f seconds\n", (double)t1 / CLOCKS_PER_SEC);
	cudaFree(accels);
	cudaFree(values);
	freeDeviceMemory();
	freeHostMemory();
}
