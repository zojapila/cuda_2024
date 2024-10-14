#include "ppmIO.h"

#define TILE_WIDTH 16

//@@ INSERT CODE HERE
__global__ void blurKernel(float *out, float *in, int width, int height, int blurSize)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int blurRadius = (blurSize - 1) / 2;
	int sum = 0;

    if ((x < width - blurRadius && y < height - blurRadius) && (x  - blurRadius >= 0 && y - blurRadius >= 0))
    {
        for (int i = x - blurRadius; i <= x + blurRadius; i++) {
			for (int j = y - blurRadius; j <= y + blurRadius; j++) {
				sum += in[j*width+ i];
			}
		}
    }
	out[y*width+x] = sum / (blurSize*blurSize);
}
//@@

int main(int argc, char *argv[])
{
	// check if number of input args is correct
	if (argc != 4)
	{
		printf("Wrong number of arguments: exactly 3 arguments needed (input and output .ppm filename with blur size)\n");
		return 0;
	}

	// get blur size
	int blurSize = atoi(argv[3]);

	///////////////////////////////////////////////////////
	//@@ INSERT CODE HERE
	unsigned int width, height;
    getPPMSize(argv[1], &width, &height);
    // read input image to a host variable
    float *hostInputImageData = (float *)malloc(width * height  * sizeof(float));
    readPPM(argv[1], hostInputImageData, true);

	// allocate input and output images in the device
    float *deviceInputImageData;
    float *deviceOutputImageData;
    cudaMalloc((void **)&deviceInputImageData, width * height  * sizeof(float));
    cudaMalloc((void **)&deviceOutputImageData, width * height * sizeof(float));

    // copy image to the device
    cudaMemcpy(deviceInputImageData, hostInputImageData, width * height * sizeof(float), cudaMemcpyHostToDevice);

    //  ///////////////////////////////////////////////////////
    //  //@@ INSERT CODE HERE

    dim3 dimGrid(ceil((float)width / TILE_WIDTH), ceil((float)height / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    blurKernel<<<dimGrid, dimBlock>>>(deviceOutputImageData, deviceInputImageData, width, height, blurSize);

    //  ///////////////////////////////////////////////////////

    float *hostOutputImageData = (float *)malloc(width * height * sizeof(float));
    cudaMemcpy(hostOutputImageData, deviceOutputImageData, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    writePPM(argv[2], hostOutputImageData, width, height, 1);

    free(hostInputImageData);
    free(hostOutputImageData);
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);

	///////////////////////////////////////////////////////

	return 0;
}
