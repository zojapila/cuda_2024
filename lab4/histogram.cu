#include "histogramUtils.h"
#include <stdio.h>

#define N_LETTERS 26
#define WARPS_SIZE 32

//@@ INSERT CODE HERE
// Histogram - basic parallel implementation
__global__ void histogram_1(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
	int binWidth = ceil(26.0/nBins);
	for (int i = 0; i < size; i++)
	{
		int alphabetPosition = buffer[i] - 'a';
		if (alphabetPosition >= 0 && alphabetPosition < 26) 
		{
			histogram[alphabetPosition / binWidth]++;
		}
	}

}

//@@ INSERT CODE HERE
// Histogram - interleaved partitioning
__global__ void histogram_2(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
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
	int dimBlock = 256;
	int dimGrid = (size + dimBlock - 1) / dimBlock;
	unsigned char *d_buffer;
	unsigned int *d_histogram, *h_histogram;
	h_histogram = (unsigned int *)malloc(nBins * sizeof(unsigned int));
	cudaMemset(h_histogram, 0, nBins * sizeof(unsigned int));
	cudaMalloc((void **)&d_buffer, size * sizeof(unsigned char));
	cudaMalloc((void **)&d_histogram, nBins * sizeof(unsigned int));
	cudaMemcpy(d_buffer, h_buffer, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemset(d_histogram, 0, nBins * sizeof(unsigned int));
	histogram_1<<<dimGrid,dimBlock>>>(d_buffer, size, d_histogram, nBins);
	cudaDeviceSynchronize();

	cudaMemcpy(h_histogram, d_histogram, nBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	printf("Histogram:\n");
	for (int i = 0; i < nBins; i++)
	{
		printf("Bin %d: %u\n", i, h_histogram[i]);
	}
	// Zwolnienie pamięci
	free(h_buffer);
	free(h_histogram);
	cudaFree(d_buffer);
	cudaFree(d_histogram);
	///////////////////////////////////////////////////////



	return 0;
}