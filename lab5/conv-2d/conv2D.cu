#include "ppmIO.h"

#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16
#define MAX_MASK_SIZE 15

__constant__ float c_M[MAX_MASK_SIZE * MAX_MASK_SIZE];

__global__ void basicConvolution2D(float *N, float *P, const int height, const int width, const int maskSize)
{
    //@@ INSERT CODE HERE
}

__global__ void tiledConvolution2D(float *N, float *P, const int height, const int width, const int maskSize)
{
    //@@ INSERT CODE HERE
}

void launchBasicConvolution2D(float *h_N, float *h_P, const int height, const int width, const int maskSize)
{
    //@@ INSERT CODE HERE
}

void launchTiledConvolution2D(float *h_N, float *h_P, const int height, const int width, const int maskSize)
{
    //@@ INSERT CODE HERE
}

int main(int argc, char *argv[])
{

    // check if number of input args is correct: input filename + mask size
    if (argc != 3)
    {
        printf("Wrong number of arguments: exactly 2 arguments needed (input filename and mask size)\n");
        return 1;
    }

    char outBasicName[] = "out_basic.ppm";
    char outTiledName[] = "out_tiled.ppm";

    // read image size
    unsigned int height, width;
    getPPMSize(argv[1], &width, &height);
    printf("Input image size (H x W): %d x %d\n", height, width);

    // load image
    float *img = (float *)malloc(height * width * sizeof(float));
    readPPM(argv[1], img, 1);

    // create mask
    const int maskSize = atoi(argv[2]) <= MAX_MASK_SIZE ? atoi(argv[2]) : MAX_MASK_SIZE;
    float M[maskSize * maskSize];
    for (int i = 0; i < maskSize * maskSize; ++i)
    {
        M[i] = 1.0f / (maskSize * maskSize);
    }
    cudaMemcpyToSymbol(c_M, M, maskSize * maskSize * sizeof(float));

    // run kernels
    float *P = (float *)malloc(height * width * sizeof(float));

    launchBasicConvolution2D(img, P, height, width, maskSize);
    writePPM(outBasicName, P, width, height, 1);

    launchTiledConvolution2D(img, P, height, width, maskSize);
    writePPM(outTiledName, P, width, height, 1);

    free(img);

    return 0;
}
