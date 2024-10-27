#include "histogramUtils.h"
#include <stdio.h>

#define N_LETTERS 26

//@@ INSERT CODE HERE
// Histogram - basic parallel implementation
__global__ void histogram_1(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
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

	///////////////////////////////////////////////////////

	free(h_buffer);

	return 0;
}
