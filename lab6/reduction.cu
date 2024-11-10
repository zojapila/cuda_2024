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
    __shared__ float shared[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < width) 
    {
        shared[tid] = input[idx];
    }
    else
    {
        shared[tid] = 0;
    }
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        if (tid % (2 * offset) == 0) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

__global__ void reductionKernelOp(float *input, float *output, int width)
{
    //@@ INSERT CODE HERE
    __shared__ float shared[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < width) 
    {
        shared[tid] = input[idx];
    }
    else
    {
        shared[tid] = 0;
    }
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

float launchReductionKernelBasic(float *h_input, int width)
{
     int numBlocks = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *d_input, *d_output;
    cudaMalloc(&d_input, width * sizeof(float));
    cudaMalloc(&d_output, numBlocks * sizeof(float));

    cudaMemcpy(d_input, h_input, width * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);

    reductionKernelBasic<<<dimGrid, dimBlock>>>(d_input, d_output, width);
    cudaDeviceSynchronize();

    float *h_output = (float*)malloc(numBlocks * sizeof(float));
    cudaMemcpy(h_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    float totalSum = 0;
    for (int i = 0; i < numBlocks; i++) {
        totalSum += h_output[i];
    }

    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return totalSum;
    //@@ INSERT CODE HERE
}

float launchReductionKernelOp(float *h_input, int width)
{
    //@@ INSERT CODE HERE
     int numBlocks = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *d_input, *d_output;
    cudaMalloc(&d_input, width * sizeof(float));
    cudaMalloc(&d_output, numBlocks * sizeof(float));

    cudaMemcpy(d_input, h_input, width * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);

    reductionKernelOp<<<dimGrid, dimBlock>>>(d_input, d_output, width);
    cudaDeviceSynchronize();

    float *h_output = (float*)malloc(numBlocks * sizeof(float));
    cudaMemcpy(h_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    float totalSum = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        totalSum += h_output[i];
    }

    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return totalSum;
    
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