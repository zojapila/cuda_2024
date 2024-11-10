#include "reductionFiles.h"

#include <stdio.h>

#define BLOCK_SIZE 128

float reductionSequential(float *input, int width)
{
    float sum = 0.0f;
    for (int i = 0; i < width; ++i)
    {
        sum += input[i];
    }

    return sum;
}

__global__ void reductionKernelBasic(float *input, float *output, int width)
{
    //@@ INSERT CODE HERE
}

__global__ void reductionKernelOp(float *input, float *output, int width)
{
    //@@ INSERT CODE HERE
}

float launchReductionKernelBasic(float *h_input, int width)
{
    //@@ INSERT CODE HERE
}

float launchReductionKernelOp(float *h_input, int width)
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
    float refVal = reductionSequential(inputData, inputSize);
    printf("Reference output: %.2f\n", refVal);

    // launch basic kernel
    float outputBasic = launchReductionKernelBasic(inputData, inputSize);
    if (refVal == outputBasic)
    {
        printf("Basic Kernel OK!\n");
    }
    else
    {
        printf("Basic Kernel FAIL! Output: %.2f\n", outputBasic);
    }

    // launch optimised kernel
    float outputOp = launchReductionKernelOp(inputData, inputSize);
    if (refVal == outputOp)
    {
        printf("Optimised Kernel OK!\n");
    }
    else
    {
        printf("Optimised Kernel FAIL! Output: %.2f\n", outputOp);
    }

    // write output data
    writeData("outBasic.txt", &outputBasic, 1);
    writeData("outOp.txt", &outputOp, 1);

    // clean
    free(inputData);

    return 0;
}