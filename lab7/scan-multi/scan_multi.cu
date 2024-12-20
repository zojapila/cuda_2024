#include "scanFiles.h"

#include <stdio.h>

#define SECTION_SIZE 512

void scanSequential(float *output, float *input, int width)
{
    float accumulator = input[0];
    output[0] = accumulator;
    for (int i = 1; i < width; ++i)
    {
        accumulator += input[i];
        output[i] = accumulator;
    }
}

__global__ void scanKernelX(float *Y, float *S, float *X, int width)
{
    //@@ INSERT CODE HERE
}

__global__ void scanKernelS(float *S, int width)
{
    //@@ INSERT CODE HERE
}

__global__ void updateYKernel(float *Y, float *S, int widthY)
{
    //@@ INSERT CODE HERE
}

void launchScanKernel(float *h_Y, float *h_X, int width)
{
    //@@ INSERT CODE HERE
}

int main(int argc, char *argv[])
{

    // check if number of input args is correct: input and output image filename
    if (argc != 2)
    {
        printf("Wrong number of arguments: exactly 1 arguments needed (input .txt filename)\n");
        return 1;
    }

    // read data
    int inputSize;
    float *inputData = NULL;
    int status = readData(argv[1], &inputData, &inputSize);
    if (status == NO_FILE)
    {
        fprintf(stderr, "%s: No such file or directory.\n", argv[1]);
        return 2;
    }
    else if (status == NO_MEMO)
    {
        fprintf(stderr, "Cannot allocate memory for the input data.\n");
        return 3;
    }

    // reference output
    float *outputRef = (float *)malloc(inputSize * sizeof(float));
    scanSequential(outputRef, inputData, inputSize);

    // launch kernel
    float *outputScan = (float *)malloc(inputSize * sizeof(float));
    launchScanKernel(outputScan, inputData, inputSize);

    // check results
    int nErr = 0;
    for (int i = 0; i < inputSize; ++i)
    {
        if (outputRef[i] != outputScan[i])
        {
            nErr++;
        }
    }
    if (nErr == 0)
    {
        printf("Scan Kernel OK!\n");
    }
    else
    {
        printf("Scan Kernel FAIL! %d/%d errors detected.\n", nErr, inputSize);
    }

    // write output data
    writeData("outSequential.txt", outputRef, inputSize);
    writeData("outParallel.txt", outputScan, inputSize);

    // clean
    free(inputData);
    free(outputRef);
    free(outputScan);

    return 0;
}