#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define RND_SEED 13 // for tests reproducibility
#define TILE_WIDTH 16

int createInputs(float **A, float **x, int size)
{
	// input test
	if (size <= 0)
	{
		fprintf(stderr, "Size must be greater than 0.\n");
		return 1;
	}

	// allocate memory
	*A = (float *)malloc(size * size * sizeof(float));
	if (*A == NULL)
	{
		fprintf(stderr, "Cannot allocate memory for matrix A.\n");
		return 2;
	}

	*x = (float *)malloc(size * sizeof(float));
	if (*x == NULL)
	{
		fprintf(stderr, "Cannot allocate memory for vector x.\n");
		return 2;
	}

	// fill with pseudo-random values
	srand(RND_SEED);
	for (int i = 0; i < size * size; ++i)
	{
		(*A)[i] = rand();
		if (i < size)
		{
			(*x)[i] = rand();
		}
	}

	return 0;
}

//@@ INSERT CODE HERE
__global__ void multMatrixVector(float *b, float *A, float *x, unsigned int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	float elem = 0.0;

	if (i < size) 
	{
		for (int idx = 0; idx < size; idx++) 
		{
			elem += A[i * size + idx] * x[idx];
		}
		b[i] = elem;
	}
}

int main(int argc, char **argv)
{
	// check if number of input args is correct
	if (argc != 2)
	{
		printf("Wrong number of arguments: exactly 1 argument needed (vector length)\n");
		return 0;
	}
	int length = atoi(argv[1]);

	// create input data
	float *h_A = NULL;
	float *h_x = NULL;
	int status = createInputs(&h_A, &h_x, length);
	if (status != 0)
	{
		return status;
	}

	///////////////////////////////////////////////////////
	//@@ INSERT CODE HERE
	float *d_A, *d_x, *d_b;
	cudaMalloc((void **) &d_A, sizeof(float)*length*length);
	cudaMalloc((void **) &d_x, sizeof(float)*length);
	cudaMalloc((void **) &d_b, sizeof(float)*length);
	cudaMemcpy(d_A, h_A, length*length*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, h_x, length*sizeof(float), cudaMemcpyHostToDevice);
	float *h_b = (float *)malloc(sizeof(float)*length);
	dim3 dimGrid(ceil((float)length / TILE_WIDTH), ceil((float)length / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	//kenrel
	multMatrixVector<<<dimGrid, dimBlock>>>(d_b, d_A, d_x, length);

	cudaMemcpy(h_b, d_b, sizeof(float)*length, cudaMemcpyDeviceToHost);
	cudaFree(d_A); cudaFree(d_b); cudaFree(d_x);



	// // save output values to file
	FILE *fp = fopen("mult_mv_out.txt", "w");
	if (fp == NULL)
	{
		fprintf(stderr, "Cannot open output file!\n");
	}
	else
	{
		for (int i = 0; i < length; ++i)
		{
			fprintf(fp, "%.0f ", h_b[i]);
		}
		fclose(fp);
	}

	///////////////////////////////////////////////////////

	return 0;
}
