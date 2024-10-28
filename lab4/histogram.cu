#include "histogramUtils.h"
#include <stdio.h>
#include <iostream>


#define N_LETTERS 26
//@@ INSERT CODE HERE
// Histogram - basic parallel implementation
__global__ void histogram_1(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
	int binWidth = ceil(26.0/nBins);
	// int batch_size = ceil(size / (gridDim.x * blockDim.x));
	int batch_size = (size + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x); 
	long idx = threadIdx.x * batch_size + blockIdx.x * blockDim.x * batch_size;
	long endIdx = min(idx + batch_size, size);
	if (idx <= size)
	{
		for (long pos = idx; pos < endIdx; pos++) 
		{
		int alphabetPosition = buffer[pos] - 'a';
		if (alphabetPosition >= 0 && alphabetPosition < 26) 
		{
			atomicAdd(&histogram[alphabetPosition / binWidth], 1);
		}
		}
	}
}

//@@ INSERT CODE HERE
// Histogram - interleaved partitioning
__global__ void histogram_2(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
	int binWidth = ceil(26.0/nBins);
	int threadsNum = gridDim.x * blockDim.x;
	// int batch_size = ceil(size / (gridDim.x * blockDim.x));
	long idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx <= size)
	{
		for (long pos = idx; pos < size; pos += threadsNum) 
		{
		int alphabetPosition = buffer[pos] - 'a';
		if (alphabetPosition >= 0 && alphabetPosition < 26) 
		{
			atomicAdd(&histogram[alphabetPosition / binWidth], 1);
		}
		}
	}


}

//@@ INSERT CODE HERE
// Histogram - interleaved partitioning + privatisation
__global__ void histogram_3(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
}

//@@ EXTRA: INSERT CODE HERE
// Extra: Histogram - interleaved partitioning + privatisation + aggregation
__global__ void histogram_4(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
}

int main(int argc, char **argv)
{
	// check if number of input args is correct: input text filename
	if (argc < 2 || argc > 3)
	{
		printf("Wrong number of arguments! Expecting 1 mandatory argument (input .txt filename) and 1 optional argument (number of bins). \n");
		return 0;
	}

	// read input string
	long size = getCharsNo(argv[1]) + 1;
	unsigned char *h_buffer = (unsigned char *)malloc(size * sizeof(unsigned char));
	readFile(argv[1], size, h_buffer);
	printf("Input string size: %ld\n", size);

	// set number of bins
	int nBins = 7;
	if (argc > 2)
	{
		int inBinsVal = atoi(argv[2]);
		if (inBinsVal > 0 && inBinsVal <= N_LETTERS)
		{
			nBins = inBinsVal;
		}
		else
		{
			fprintf(stderr, "Incorrect input number of bins: %d. Proceeding with default value: %d.\n", inBinsVal, nBins);
		}
	}

	///////////////////////////////////////////////////////
	//@@ INSERT CODE HERE
	// HISTOGRAM 1
	// dim3 dimBlock (256,1,1);
	// dim3 dimGrid ((size + 256 * 32 - 1 ) / (256 * 32),1,1);
	// unsigned char *d_buffer;
	// unsigned int *d_histogram, *h_histogram;

	// h_histogram = (unsigned int *)malloc(nBins * sizeof(unsigned int));
	// memset(h_histogram, 0, nBins * sizeof(unsigned int));  // Użycie memset zamiast cudaMemset

	// cudaMalloc((void **)&d_buffer, size * sizeof(unsigned char));
	// cudaMalloc((void **)&d_histogram, nBins * sizeof(unsigned int));
	// cudaMemcpy(d_buffer, h_buffer, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	// cudaMemset(d_histogram, 0, nBins * sizeof(unsigned int)); // Zerowanie histogramu na urządzeniu

	// histogram_1<<<dimGrid,dimBlock>>>(d_buffer, size, d_histogram, nBins);
	// cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
	// cudaDeviceSynchronize();

	// cudaMemcpy(h_histogram, d_histogram, nBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	// printf("Histogram:\n");
	// for (int i = 0; i < nBins; i++)
	// {
	// 	printf("Bin %d: %u\n", i, h_histogram[i]);
	// }
	// char* filename = (char *)"histogram_out.txt";
	// writeFile(filename, h_histogram, nBins, 26);
	// // Zwolnienie pamięci
	// free(h_buffer);
	// free(h_histogram);
	// cudaFree(d_buffer);
	// cudaFree(d_histogram);
	////////////////////////////////////////////////////////////////
	// HISTOGRAM 2
	dim3 dimBlock2 (256,1,1);
	dim3 dimGrid2 ((size + 256 - 1 ) / (256),1,1);
	unsigned char *d_buffer2;
	unsigned int *d_histogram2, *h_histogram2;

	h_histogram2 = (unsigned int *)malloc(nBins * sizeof(unsigned int));
	memset(h_histogram2, 0, nBins * sizeof(unsigned int));  // Użycie memset zamiast cudaMemset

	cudaMalloc((void **)&d_buffer2, size * sizeof(unsigned char));
	cudaMalloc((void **)&d_histogram2, nBins * sizeof(unsigned int));
	cudaMemcpy(d_buffer2, h_buffer, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemset(d_histogram2, 0, nBins * sizeof(unsigned int)); // Zerowanie histogramu na urządzeniu

	histogram_1<<<dimGrid2,dimBlock2>>>(d_buffer2, size, d_histogram2, nBins);
	cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
	cudaDeviceSynchronize();

	cudaMemcpy(h_histogram2, d_histogram2, nBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	printf("Histogram:\n");
	for (int i = 0; i < nBins; i++)
	{
		printf("Bin %d: %u\n", i, h_histogram2[i]);
	}
	char* filename = (char *)"histogram_out.txt";
	writeFile(filename, h_histogram2, nBins, 26);
	// Zwolnienie pamięci
	free(h_buffer);
	free(h_histogram2);
	cudaFree(d_buffer2);
	cudaFree(d_histogram2);
	///////////////////////////////////////////////////////

	return 0;
}