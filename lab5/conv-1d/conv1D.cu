#include "convFiles.h"
#include <stdio.h>

#define TILE_SIZE 1024
#define MAX_MASK_WIDTH 15

__constant__ float c_M[MAX_MASK_WIDTH];

__global__ void basicConvolution1D(float *N, float *M, float *P, const int width, const int maskWidth)
{
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int start_idx = max(0, idx - maskWidth/2) - idx + maskWidth/2;
    float sum = 0;
    for (int i = max(0, idx - maskWidth/2); i <= min(width, idx + maskWidth/2); i++) 
    {
        sum += N[i] * M[start_idx];
        start_idx ++;
    }
    if (idx < width) P[idx] = sum;
}

__global__ void tiledConvolution1D(float *N, float *P, const int width, const int maskWidth)
{
    __device__ __shared__ float shared[MAX_MASK_WIDTH+TILE_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_shared = idx - (maskWidth/2);
    if (idx_shared >= 0 && idx_shared < width) 
    {
        shared[threadIdx.x] = N[idx_shared];
        // printf("%f ", N[idx_shared]);
    }
    else
    {
        shared[threadIdx.x] = 0;
    }

    idx_shared = idx + (maskWidth/2) + TILE_SIZE;
    if (idx_shared >= 0 && idx_shared < width && (MAX_MASK_WIDTH > threadIdx.x)) 
    {
        // printf("pomidor");
        shared[threadIdx.x + TILE_SIZE] = N[idx_shared];
        // printf("%f ", N[idx_shared]);
    }
    else if (MAX_MASK_WIDTH > threadIdx.x)
    {
        shared[threadIdx.x + TILE_SIZE] = 0;
    }
    __syncthreads();

    float sum = 0;
    int start = 0;
    for (int i=threadIdx.x; i < threadIdx.x + maskWidth; i++) 
    {
        sum += shared[i] * c_M[start];
        // printf("%f", c_M[start]);
        start++;
        // printf("%f ", sum);
    }
    if(idx<width) 
    {
        // printf("ziemniak %f ", sum);
        P[idx] = sum;
    }
}

void launchBasicConvolution1D(float *h_N, float *h_M, float *h_P, const int width, const int maskWidth)
{
    //@@ INSERT CODE HERE
    dim3 dimGrid(ceil((float)width/TILE_SIZE), 1, 1);
    dim3 dimBlock(TILE_SIZE,1,1);

    float *d_N, *d_M, *d_P;

    cudaMalloc(&d_M, maskWidth * sizeof(float));
    cudaMalloc(&d_N, width * sizeof(float));
    cudaMalloc(&d_P, width * sizeof(float));

    cudaMemcpy(d_N, h_N, width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, maskWidth*sizeof(float), cudaMemcpyHostToDevice);
    // printf("twoja stara");

    basicConvolution1D<<<dimGrid, dimBlock>>>(d_N, d_M, d_P, width, maskWidth);
    cudaMemcpy(h_P, d_P, width * sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f",h_P[0]);
	cudaFree(d_P);
	cudaFree(d_M);
    cudaFree(d_N);
}

void launchTiledConvolution1D(float *h_N, float *h_P, const int width, const int maskWidth)
{
    //@@ INSERT CODE HERE
    dim3 dimGrid(ceil((float)width/TILE_SIZE), 1, 1);
    dim3 dimBlock(TILE_SIZE,1,1);

    float *d_N, *d_P;

    cudaMalloc(&d_N, width * sizeof(float));
    cudaMalloc(&d_P, width * sizeof(float));

    cudaMemcpy(d_N, h_N, width*sizeof(float), cudaMemcpyHostToDevice);

    tiledConvolution1D<<<dimGrid, dimBlock>>>(d_N, d_P, width, maskWidth);
    cudaMemcpy(h_P, d_P, width * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_P);
    cudaFree(d_N);
}

int main(int argc, char *argv[])
{
    // check if number of input args is correct: input filename
    if (argc != 2)
    {
        printf("Wrong number of arguments: exactly 1 argument needed (input .txt filename)\n");
        return 1;
    }

    // output names
    char nameOutBasic[] = "out_1D_basic.txt";
    char nameOutTiled[] = "out_1D_tiled.txt";

    // read sizes
    int width, maskWidth;
    int status = getSizes1D(argv[1], &maskWidth, &width);
    if (status == NO_FILE)
    {
        printf("%s: No such file or directory,\n", argv[1]);
        return 2;
    }

    // read data
    float *N = (float *)malloc(width * sizeof(float));
    float *M = (float *)malloc(maskWidth * sizeof(float));
    getValues1D(argv[1], M, N);

    // for the output data
    float *P = (float *)malloc(width * sizeof(float));

    // basic kernel
    launchBasicConvolution1D(N, M, P, width, maskWidth);
    writeData1D(nameOutBasic, P, width);
    // printf("dupa");
    // free(P);
    cudaMemcpyToSymbol(c_M, M, maskWidth * sizeof(float));
    // tiled kernel
    //@@ INSERT CODE HERE
    launchTiledConvolution1D(N, P, width, maskWidth);
    writeData1D(nameOutTiled, P, width);

    free(P);
    free(N);
    free(M);

    return 0;
}
