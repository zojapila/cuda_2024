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

///////////////////////////////////////////////////////////////////////

#include <iostream>
#include <iomanip> //for precision
#include <memory>
#include <cuda.h>

#include "CLI11.hpp" //argument parser
#include "CudaMemoryManager.hpp"

constexpr unsigned int TILE_WIDTH = 32;

// Compute C = A * B general matrix-matrix multiply
__global__ void standardMatrixMult(float *A, float *B, float *C, int numARows,
                                   int numAColumns, int numBRows, int numBColumns) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numARows && col < numBColumns) {
        float sum = 0;
        for (int ii = 0; ii < numAColumns; ii++) {
            sum += A[row * numAColumns + ii] * B[ii * numBColumns + col];
        }
        C[row * numBColumns + col] = sum;
    }
}

//@@ INSERT CODE HERE
// Compute C = A * B tiled matrix-matrix multiply
// 1 element of C => 1 thread
// # of threads in tile: TILE_WIDTH * TILE_WIDTH
__global__ void matrixMultiply(float *A, unsigned int A_cols_num, unsigned int A_rows_num,
                               float *B, unsigned int B_cols_num, unsigned int B_rows_num,
                               float *C, unsigned int C_cols_num, unsigned int C_rows_num) {

    //Position in matrix C
    const unsigned int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const unsigned int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float p_value = 0.0;


    __device__ __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __device__ __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    //Load memory from A und B & process it
    auto tmp = fdividef(A_cols_num, TILE_WIDTH); //HUGE PROBLEM WITH DIVIDE "/", RETURNS 0!!!
    printf("Dzielenie: %f\n", float(A_cols_num)/TILE_WIDTH);
    for (int phase = 0; phase < int(ceilf(tmp)); phase++) {

        //read rows of A (moving col)
        if (threadIdx.x + phase * TILE_WIDTH >= A_cols_num || row >= A_rows_num) {
            ds_A[threadIdx.y][threadIdx.x] = 0.0;
        } else {
            ds_A[threadIdx.y][threadIdx.x] = A[(row * A_cols_num) + (phase * TILE_WIDTH + threadIdx.x)];
        }


        if ((threadIdx.y + phase * TILE_WIDTH) >= B_rows_num || col >= B_cols_num) {
            ds_B[threadIdx.y][threadIdx.x] = 0.0;
        } else {
            ds_B[threadIdx.y][threadIdx.x] = B[(phase * TILE_WIDTH + threadIdx.y) * B_cols_num + col];
        }

        __syncthreads();

        // Accumulate sum for C[row][col] element
        for (int cnt = 0; cnt < TILE_WIDTH; cnt++) {
            p_value += ds_A[threadIdx.y][cnt] * ds_B[cnt][threadIdx.x];
        }

        __syncthreads();
    }

    //Write p_value to output matrix
    if (row < C_rows_num && col < C_cols_num) {
        C[row * C_cols_num + col] = p_value;
    }
}
//

// Compute C = A * B tiled matrix-matrix multiply - Less detailed version
// 2 elements of C => 1 thread
// # of threads in tile: TILE_WIDTH * TILE_WIDTH
__global__ void matrixMultiply2(float *A, unsigned int A_cols_num, unsigned int A_rows_num,
                                float *B, unsigned int B_cols_num, unsigned int B_rows_num,
                                float *C, unsigned int C_cols_num, unsigned int C_rows_num) {

    //Position in matrix C
    const unsigned int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const unsigned int col = (blockIdx.x * 2) * TILE_WIDTH + threadIdx.x;
    const unsigned int col2 = (blockIdx.x * 2 + 1) * TILE_WIDTH + threadIdx.x;
    float p_value = 0.0;
    float p_value2 = 0.0;


    __device__ __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __device__ __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH * 2];

    //Load memory from A und B & process it
    auto tmp = fdividef(A_cols_num, TILE_WIDTH); //HUGE PROBLEM WITH DIVIDE "/", RETURNS 0!!!
    for (int phase = 0; phase < int(ceilf(tmp)); phase++) {

        //read rows of A (moving col)
        if (threadIdx.x + phase * TILE_WIDTH >= A_cols_num || row >= A_rows_num) {
            ds_A[threadIdx.y][threadIdx.x] = 0.0;
        } else {
            ds_A[threadIdx.y][threadIdx.x] = A[(row * A_cols_num) + (phase * TILE_WIDTH + threadIdx.x)];
        }


        if ((threadIdx.y + phase * TILE_WIDTH) >= B_rows_num || col >= B_cols_num) {
            ds_B[threadIdx.y][threadIdx.x] = 0.0;
        } else {
            ds_B[threadIdx.y][threadIdx.x] = B[(phase * TILE_WIDTH + threadIdx.y) * B_cols_num + col];
        }

        if ((threadIdx.y + phase * TILE_WIDTH) >= B_rows_num || col2 >= B_cols_num) {
            ds_B[threadIdx.y][threadIdx.x + TILE_WIDTH] = 0.0;
        } else {
            ds_B[threadIdx.y][threadIdx.x + TILE_WIDTH] = B[(phase * TILE_WIDTH + threadIdx.y) * B_cols_num + col2];
        }

        __syncthreads();

        // Accumulate sum for C[row][col] element
        for (int cnt = 0; cnt < TILE_WIDTH; cnt++) {
            p_value += ds_A[threadIdx.y][cnt] * ds_B[cnt][threadIdx.x];
            p_value2 += ds_A[threadIdx.y][cnt] * ds_B[cnt][threadIdx.x + TILE_WIDTH];
        }

        __syncthreads();
    }

    //Write p_value to output matrix
    if (row < C_rows_num && col < C_cols_num) {
        C[row * C_cols_num + col] = p_value;
    }

    if (row < C_rows_num && col2 < C_cols_num) {
        C[row * C_cols_num + col2] = p_value2;
    }
}

void generateRandomFlattenMatrix(float *M, unsigned int size) {
    for (int i = 0; i < size; ++i) {
        M[i] = (rand() % 20) + 50;
    }
}

void printMatrix(const std::shared_ptr<float[]> M,
                 const unsigned int cols = 1,
                 const unsigned int rows = 1) {

    //TODO: implement automatic hiding
    if (cols > 10 || rows > 10) {
        std::cout << "Matrix is too big to be printed!\n";
        return;
    }

    std::cout << "[ ";
    for (int cnt_row = 0; cnt_row < rows; cnt_row++) {
        for (int cnt_col = 0; cnt_col < cols; cnt_col++) {
            std::cout << std::setprecision(2) << std::fixed << M[cnt_row * cols + cnt_col] << ",  ";
        }

        if (cnt_row + 1 != rows) {
            std::cout << "\b\n  ";
        }
    }
    std::cout << "\b\b ]\n\n";
}

int main(int argc, char **argv) {

    ///////////////////////////////////////////////////////
    //@@ INSERT CODE HERE
    CLI::App app{"CUDA Matrix Multiply using Tiling"};

    std::cout << "> Multiplying 2 matrixes \n\n";

    unsigned int A_rows = 4;
    unsigned int A_cols = 4;
    unsigned int B_rows = 4;
    unsigned int B_cols = 1;
    unsigned int C_rows = A_rows;
    unsigned int C_cols = B_cols;


    app.add_option("--a_rows", A_rows, "Number of rows in matrix A");
    app.add_option("--a_cols", A_cols, "Number of cols in matrix A");
    app.add_option("--b_rows", B_rows, "Number of rows in matrix B");
    app.add_option("--b_cols", B_cols, "Number of cols in matrix B");
    app.add_option("--c_rows", C_rows, "Number of rows in matrix C");
    app.add_option("--c_cols", C_cols, "Number of cols in matrix C");

    CLI11_PARSE(app, argc, argv);
    std::cout << "> A dims are " << A_rows << "x" << A_cols << "\n";
    std::cout << "> B dims are " << B_rows << "x" << B_cols << "\n";
    std::cout << "> C dims are " << A_rows << "x" << B_cols << "\n";

    if (A_cols != B_rows || C_cols != B_cols || C_rows != A_rows) {
        std::cout << "> Validation failed: Can't perform matrix multiplication C=A*B.\n";
        return 0;
    }

    auto cuda_A = CUDAMemoryManager<float>(A_rows * A_cols);
    auto cuda_B = CUDAMemoryManager<float>(B_rows * B_cols);
    auto cuda_C = CUDAMemoryManager<float>(C_rows * C_cols);
    std::shared_ptr<float[]> mat_A(new float[A_rows * A_cols]);
    std::shared_ptr<float[]> mat_B(new float[B_rows * B_cols]);
    std::shared_ptr<float[]> mat_C(new float[C_rows * C_cols]);

    generateRandomFlattenMatrix(mat_A.get(), A_rows * A_cols);
    generateRandomFlattenMatrix(mat_B.get(), B_rows * B_cols);

    std::cout << "Generated A =\n";
    printMatrix(mat_A, A_cols, A_rows);

    std::cout << "Generated B =\n";
    printMatrix(mat_B, B_cols, B_rows);


    // copy matrxies to the device
    cuda_A.copy_to_device(mat_A.get());
    cuda_B.copy_to_device(mat_B.get());

    // ------------------------------------------
    // RUN THE KERNELS
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(ceil((float) B_cols / TILE_WIDTH),
                 ceil((float) A_rows / TILE_WIDTH), 1);

    // VERSION 1
    std::cout << "> C=A*B using tailing method\n";
    matrixMultiply<<<dimGrid, dimBlock>>>(cuda_A.get(), A_cols, A_rows,
                                          cuda_B.get(), B_cols, B_rows,
                                          cuda_C.get(), C_cols, C_rows);
    cudaDeviceSynchronize();
    cuda_C.copy_to_host(mat_C.get());
    std::cout << "Result C =\n";
    printMatrix(mat_C, C_cols, C_rows);

    // VERSION 2
    std::cout << "> C=A*B using without tailing method\n";
    standardMatrixMult<<<dimGrid, dimBlock>>>(cuda_A.get(), cuda_B.get(), cuda_C.get(),
                                              A_rows, A_cols, B_rows, B_cols);
    cudaDeviceSynchronize();
    cuda_C.copy_to_host(mat_C.get());
    std::cout << "Result C =\n";
    printMatrix(mat_C, C_cols, C_rows);


    // VERSION 1
    dim3 dimBlock2(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid2(ceil((float) B_cols / (TILE_WIDTH * 2)),
                  ceil((float) A_rows / TILE_WIDTH), 1);

    std::cout << "> C=A*B using less detailed tailing method\n";
    matrixMultiply2<<<dimGrid2, dimBlock2>>>(cuda_A.get(), A_cols, A_rows,
                                             cuda_B.get(), B_cols, B_rows,
                                             cuda_C.get(), C_cols, C_rows);
    cudaDeviceSynchronize();
    cuda_C.copy_to_host(mat_C.get());
    std::cout << "Result C =\n";
    printMatrix(mat_C, C_cols, C_rows);

    ///////////////////////////////////////////////////////

    return 0;
}

// Macierze 5000
//==22676== Profiling application: /home/mad/docs/s9_cuda/lab3/Lab-03/EX-1-Tiling/cmake-build-debug/ex1_tiling
//==22676== Profiling result:
//Type  Time(%)      Time     Calls       Avg       Min       Max  Name
//GPU activities:   68.66%  3.12159s         1  3.12159s  3.12159s  3.12159s  standardMatrixMult(float*, float*, float*, int, int, int, int)
//                  29.54%  1.34324s         1  1.34324s  1.34324s  1.34324s  matrixMultiply(float*, unsigned int, unsigned int, float*, unsigned int, unsigned int, float*, unsigned int, unsigned int)
//                   1.27%  57.864ms         2  28.932ms  12.446ms  45.419ms  [CUDA memcpy DtoH]
//                   0.52%  23.757ms         2  11.878ms  11.808ms  11.949ms  [CUDA memcpy HtoD]
//API calls:   95.19%  4.46490s         2  2.23245s  1.34330s  3.12160s  cudaDeviceSynchronize
//2.73%  127.82ms         3  42.608ms  205.93us  127.41ms  cudaMalloc
//1.76%  82.739ms         4  20.685ms  11.906ms  46.256ms  cudaMemcpy
//0.30%  13.844ms         3  4.6148ms  244.35us  6.8028ms  cudaFree
//0.01%  372.27us         1  372.27us  372.27us  372.27us  cuDeviceGetName
//0.01%  364.29us       101  3.6060us     194ns  209.58us  cuDeviceGetAttribute
//0.01%  236.35us         1  236.35us  236.35us  236.35us  cuDeviceTotalMem
//0.00%  61.207us         2  30.603us  28.427us  32.780us  cudaLaunchKernel
//0.00%  10.935us         1  10.935us  10.935us  10.935us  cuDeviceGetPCIBusId
//0.00%  2.5560us         3     852ns     362ns  1.7860us  cuDeviceGetCount
//0.00%  1.6050us         2     802ns     279ns  1.3260us  cuDeviceGet
//0.00%     372ns         1     372ns     372ns     372ns  cuDeviceGetUuid

// Macierze 18000 (dla 20000 potrzeba +4,5 GB pamieci, moje GPU ma tylko 4GB)

//==22920== NVPROF is profiling process 22920, command: /home/mad/docs/s9_cuda/lab3/Lab-03/EX-1-Tiling/cmake-build-debug/ex1_tiling
//==22920== Profiling application: /home/mad/docs/s9_cuda/lab3/Lab-03/EX-1-Tiling/cmake-build-debug/ex1_tiling
//==22920== Profiling result:
//Type  Time(%)      Time     Calls       Avg       Min       Max  Name
//GPU activities:   77.39%  218.891s         1  218.891s  218.891s  218.891s  standardMatrixMult(float*, float*, float*, int, int, int, int)
//                  22.19%  62.7732s         1  62.7732s  62.7732s  62.7732s  matrixMultiply(float*, unsigned int, unsigned int, float*, unsigned int, unsigned int, float*, unsigned int, unsigned int)
//                   0.30%  844.46ms         2  422.23ms  167.41ms  677.05ms  [CUDA memcpy DtoH]
//                   0.12%  329.72ms         2  164.86ms  160.73ms  168.99ms  [CUDA memcpy HtoD]
//     API calls:   99.48%  281.664s         2  140.832s  62.7733s  218.891s  cudaDeviceSynchronize
//                   0.42%  1.17561s         4  293.90ms  160.81ms  678.15ms  cudaMemcpy
//                   0.06%  177.21ms         3  59.071ms  1.7981ms  87.865ms  cudaFree
//                   0.04%  117.69ms         3  39.231ms  1.6720ms  114.32ms  cudaMalloc
//                   0.00%  198.53us         1  198.53us  198.53us  198.53us  cuDeviceTotalMem
//                   0.00%  160.00us       101  1.5840us     154ns  68.132us  cuDeviceGetAttribute
//                   0.00%  62.383us         2  31.191us  28.593us  33.790us  cudaLaunchKernel
//                   0.00%  30.300us         1  30.300us  30.300us  30.300us  cuDeviceGetName
//                   0.00%  14.177us         1  14.177us  14.177us  14.177us  cuDeviceGetPCIBusId
//                   0.00%  6.5130us         2  3.2560us     188ns  6.3250us  cuDeviceGet
//                   0.00%  1.5750us         3     525ns     219ns  1.1160us  cuDeviceGetCount
//                   0.00%     325ns         1     325ns     325ns     325ns  cuDeviceGetUuid


// ZADANIE DOTATKOWE - porownanie kerneli, macierze 5000
//==30126== Profiling result:
//Type  Time(%)      Time     Calls       Avg       Min       Max  Name
//GPU activities:   56.07%  3.12554s         1  3.12554s  3.12554s  3.12554s  standardMatrixMult(float*, float*, float*, int, int, int, int)
//                  23.99%  1.33752s         1  1.33752s  1.33752s  1.33752s  matrixMultiply(float*, unsigned int, unsigned int, float*, unsigned int, unsigned int, float*, unsigned int, unsigned int)
//                  18.23%  1.01608s         1  1.01608s  1.01608s  1.01608s  matrixMultiply2(float*, unsigned int, unsigned int, float*, unsigned int, unsigned int, float*, unsigned int, unsigned int)
//                   1.27%  70.796ms         3  23.599ms  12.436ms  45.649ms  [CUDA memcpy DtoH]
//                   0.44%  24.590ms         2  12.295ms  12.066ms  12.525ms  [CUDA memcpy HtoD]

// POROWANIE KERNELI, MACIERZE 18500
//==30977== Profiling result:
//Type  Time(%)      Time     Calls       Avg       Min       Max  Name
//GPU activities:   67.82%  256.836s         1  256.836s  256.836s  256.836s  standardMatrixMult(float*, float*, float*, int, int, int, int)
//                  18.14%  68.6897s         1  68.6897s  68.6897s  68.6897s  matrixMultiply(float*, unsigned int, unsigned int, float*, unsigned int, unsigned int, float*, unsigned int, unsigned int)
//                  13.66%  51.7463s         1  51.7463s  51.7463s  51.7463s  matrixMultiply2(float*, unsigned int, unsigned int, float*, unsigned int, unsigned int, float*, unsigned int, unsigned int)
//                   0.29%  1.11050s         3  370.17ms  168.49ms  770.71ms  [CUDA memcpy DtoH]
//                   0.09%  335.15ms         2  167.58ms  165.17ms  169.98ms  [CUDA memcpy HtoD]