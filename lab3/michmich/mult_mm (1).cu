#include <stdio.h>
#include "matUtils.h"

#define RND_SEED 13 // for tests reproducibility
#define TILE_WIDTH 16



// Compute C = A * B general matrix-matrix multiply
__global__ void standardMatrixMult(float *A, float *B, float *C, int numARows,
                                   int numAColumns, int numBRows, int numBColumns)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numARows && col < numBColumns)
    {
        float sum = 0;
        for (int ii = 0; ii < numAColumns; ii++)
        {
            sum += A[row * numAColumns + ii] * B[ii * numBColumns + col];
        }
        C[row * numBColumns + col] = sum;
    }
}

#define TILE_WIDTH 16

//@@ INSERT CODE HERE
// Compute C = A * B tiled matrix-matrix multiply
__global__ void matrixMultiply( float* C, float* A, float* B, int height_A, int width_A, int width_B)
{
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH], ds_B[TILE_WIDTH][TILE_WIDTH];
    float sum = 0;
    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    for(int ph=0; ph<ceil((float)width_A/TILE_WIDTH); ph++){
        int sid_A =  (ph*TILE_WIDTH +threadIdx.y) + (TILE_WIDTH * blockIdx.x + threadIdx.x) * width_A;
        if((ph*TILE_WIDTH +threadIdx.y) < width_A && (TILE_WIDTH * blockIdx.x + threadIdx.x)<height_A){
            ds_A[threadIdx.x][threadIdx.y] = A[sid_A];
        }else{
            ds_A[threadIdx.x][threadIdx.y] = 0;
        }
        int sid_B = (ph*TILE_WIDTH +threadIdx.x)*width_B + ((TILE_WIDTH * blockIdx.y) + threadIdx.y);
        if((blockIdx.y*TILE_WIDTH + threadIdx.y) < width_B && (TILE_WIDTH * ph + threadIdx.x) < width_A){
            ds_B[threadIdx.x][threadIdx.y] = B[sid_B];
        }else{
            ds_B[threadIdx.x][threadIdx.y] = 0;
        }
        __syncthreads();
        for(int i = 0; i<TILE_WIDTH; i++){
            
            sum+= ds_A[threadIdx.x][i] * ds_B[i][threadIdx.y];
        }
        __syncthreads();
    }
    if(idx_x < height_A && idx_y < width_B){
        C[idx_x * width_B + idx_y ] = sum;
    }
}
//

void generateRandomFlattenMatrix(float *M, unsigned int size)
{
    for (int i = 0; i < size; ++i)
    {
        M[i] = (rand() % 20) + 50;
    }
}

int main(int argc, char **argv)
{
    // check if number of input args is correct
    if (argc < 4 || argc > 7)
    {
        printf("Wrong number of arguments: 3 mandatory arguments needed (width A, height A and width B)\n");
        printf("If 4th argument is --read then input matrix are read from the files given as 5th and 6th arguments.\n");
        printf("Example: ./mult_mm.out 5 8 13 --read ./inputA.txt ./inputB.txt");
        return 0;
    }
    int widthA = atoi(argv[1]);
    int heightA = atoi(argv[2]);
    int widthB = atoi(argv[3]);

    int readFile = 0;
    int matAsize = widthA * heightA;
    int matBsize = widthB * widthA;
    if (argc > 4)
    {
        // Check matrix A size
        int status = getMatSize(argv[5], &matAsize);
        if (status != MAT_SUCCESS)
        {
            printf("Error: getMatSize for matrix A, status: %d\n", status);
            return 0;
        }
        if (matAsize != widthA * heightA)
        {
            printf("Matrix A size mismtach: %d vs %d.\n", matAsize, widthA * heightA);
            return 0;
        }

        // Check matrix B size
        status = getMatSize(argv[6], &matBsize);
        if (status != MAT_SUCCESS)
        {
            printf("Error: getMatSize for matrix B, status: %d\n", status);
            return 0;
        }
        if (matBsize != widthB * widthA)
        {
            printf("Matrix B size mismtach: %d vs %d.\n", matBsize, widthB * widthA);
            return 0;
        }

        readFile = 1;
    }

    float *h_A = (float *)malloc(matAsize * sizeof(float));
    float *h_B = (float *)malloc(matBsize * sizeof(float));

    if (!readFile)
    {
        srand(RND_SEED);

        // Generate matrix A
        int status = generateMat("./inputA.txt", h_A, matAsize);
        if (status != MAT_SUCCESS)
        {
            printf("Error: generateMat for matrix A, status: %d\n", status);
            return 0;
        }

        // Generate matrix B
        status = generateMat("./inputB.txt", h_B, matBsize);
        if (status != MAT_SUCCESS)
        {
            printf("Error: generateMat for matrix B, status: %d\n", status);
            return 0;
        }
    }
    else
    {
        // Read matrix A
        int status = readMat(argv[5], h_A, matAsize);
        if (status != MAT_SUCCESS)
        {
            printf("Error: readMat for matrix A, status: %d\n", status);
            return 0;
        }

        // Read matrix B
        status = readMat(argv[6], h_B, matBsize);
        if (status != MAT_SUCCESS)
        {
            printf("Error: readMat for matrix B, status: %d\n", status);
            return 0;
        }
    }

    ///////////////////////////////////////////////////////
    //@@ INSERT CODE HERE

    float *deviceB;
	float *deviceC;
    float *deviceA;



	cudaMalloc(&deviceB, matBsize * sizeof(float));
	cudaMalloc(&deviceC, heightA * widthB * sizeof(float));
	cudaMalloc(&deviceA, matAsize * sizeof(float));

	cudaMemcpy(deviceA, h_A, matAsize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, h_B, matBsize * sizeof(float), cudaMemcpyHostToDevice);


    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
    dim3 dimGrid(ceil((float)heightA/ TILE_WIDTH), ceil((float)widthB/ TILE_WIDTH));

    
    standardMatrixMult<<<dimGrid, dimBlock>>>( deviceA, deviceB, deviceC, heightA , widthA, widthA, widthB);
    matrixMultiply<<<dimGrid, dimBlock>>>(deviceC, deviceA, deviceB, heightA , widthA , widthB);


    float *h_C = (float *)malloc(heightA * widthB * sizeof(float));
    cudaMemcpy(h_C,deviceC , heightA * widthB * sizeof(float), cudaMemcpyDeviceToHost);


    // save output values
    if (true)
    {
        FILE *fp = fopen("mult_mm_out.txt", "w");
        if (fp == NULL)
        {
            fprintf(stderr, "Cannot open output file!\n");
        }
        else
        {
            printf("Generating output file... ");
            for (int i = 0; i < heightA * widthB; ++i)
            {
                fprintf(fp, "%.0f ", h_C[i]);
            }
            printf("DONE! \n");
            fclose(fp);
        }
    }

    // char name[256] = "";
    // strcat(strcat(strcat(strcat(strcat(strcat(strcat(name ,"mult_mm_") , argv[1]) , "_") ,argv[2]) , "_") , argv[3]) , "_out.txt");
    // FILE *fp = fopen(name, "w");
    // if (fp == NULL)
    // {
    //     fprintf(stderr, "Cannot open output file!\n");
    // }
    // else
    // {
    //     printf("Generating output file... ");
    //     fprintf(fp, "A\n");
    //     for (int i = 0; i < matAsize; ++i)
    //     {
    //         fprintf(fp, "%.0f ", h_A[i]);
    //     }
    //     fprintf(fp, "\nB\n");
    //     for (int i = 0; i < matBsize; ++i)
    //     {
    //         fprintf(fp, "%.0f ", h_B[i]);
    //     }
    //     printf("DONE! \n");
    //     fclose(fp);
    // }


    ///////////////////////////////////////////////////////

    return 0;
}
