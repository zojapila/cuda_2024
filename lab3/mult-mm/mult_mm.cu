#include <stdio.h>
#include "matUtils.h"

#define RND_SEED 13 // for tests reproducibility

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
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows, 
                               int numAColumns, int numBRows, int numBColumns) // w instrukcji do cwiczenia nazywa sie to matrixMult()
{
    __device__ __shared__ float ds_A [TILE_WIDTH][TILE_WIDTH]; 
    __device__ __shared__ float ds_B [TILE_WIDTH][TILE_WIDTH]; 
    int row = TILE_WIDTH * blockIdx.y + threadIdx.y;
    int col = TILE_WIDTH * blockIdx.x + threadIdx.x;
    float scalar = 0;

    for (int ph=0; ph < ceil((float)numAColumns/TILE_WIDTH); ph++) 
    {
        if (row * numAColumns < numARows && ph * TILE_WIDTH + threadIdx.x < numAColumns) 
        {
            ds_A[row][ph * TILE_WIDTH + threadIdx.x] = A[row * numAColumns + ph * TILE_WIDTH + threadIdx.x];
        }
        else 
        {
            ds_A[row][ph * TILE_WIDTH + threadIdx.x] = 0;
        }
        if (row * numBColumns < numBRows && col < numBColumns) 
        {
            ds_B[row*numBColumns][col] = B[row * numBColumns + ph * TILE_WIDTH + threadIdx.x];
        }
        else 
        {
            ds_B[row*numBColumns][col] = 0;
        }
        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; i++) 
        {
            scalar += ds_A[row][i] * ds_B[i][col];
        }
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

    // // save output values
    // if (same)
    // {
    //     FILE *fp = fopen("mult_mm_out.txt", "w");
    //     if (fp == NULL)
    //     {
    //         fprintf(stderr, "Cannot open output file!\n");
    //     }
    //     else
    //     {
    //         printf("Generating output file... ");
    //         for (int i = 0; i < heightA * widthB; ++i)
    //         {
    //             fprintf(fp, "%.0f ", h_C[i]);
    //         }
    //         printf("DONE! \n");
    //         fclose(fp);
    //     }
    // }

    ///////////////////////////////////////////////////////

    return 0;
}
