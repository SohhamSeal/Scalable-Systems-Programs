/*
Write CUDA program to multiply 2 Matrices A and B and store result in C.
*/

#include <stdio.h>
#include <math.h>

#define TILE_WIDTH 2

// shared
__global__ void MatrixMulSh(float *Md, float *Nd, float *Pd, const int WIDTH)
{
    // Taking shared array to break the Matrix in Tile width and fetch them in that array per element
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // calculate thread id
    unsigned int col = TILE_WIDTH * blockIdx.x + threadIdx.x;
    unsigned int row = TILE_WIDTH * blockIdx.y + threadIdx.y;
    for (int m = 0; m < WIDTH / TILE_WIDTH; m++) // m indicate number of phase
    {
        Mds[threadIdx.y][threadIdx.x] = Md[row * WIDTH + (m * TILE_WIDTH + threadIdx.x)];
        Nds[threadIdx.y][threadIdx.x] = Nd[(m * TILE_WIDTH + threadIdx.y) * WIDTH + col];
        __syncthreads(); // for syncronizeing the threads
        // Do for tile
        for (int k = 0; k < TILE_WIDTH; k++)
            Pd[row * WIDTH + col] += Mds[threadIdx.x][k] * Nds[k][threadIdx.y];
        __syncthreads(); // for synchronizing the threads
    }
}

// main routine
int main()
{
    // create events
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    const int WIDTH = 2;
    float array1_h[WIDTH][WIDTH], array2_h[WIDTH][WIDTH], M_result_array_h[WIDTH][WIDTH];
    float *array1_d, *array2_d, *result_array_d, *M_result_array_d; // device array
    int i, j;

    // input in host array
    for (i = 0; i < WIDTH; i++)
    {
        for (j = 0; j < WIDTH; j++)
        {
            array1_h[i][j] = i + j;
            array2_h[i][j] = i + j;
        }
    }

    // create device array cudaMalloc ( (void **)&array_name, sizeofmatrixinbytes) ;
    cudaMalloc((void **)&array1_d, WIDTH * WIDTH * sizeof(int));
    cudaMalloc((void **)&array2_d, WIDTH * WIDTH * sizeof(int));

    // copy host array to device array; cudaMemcpy ( dest , source , WIDTH , direction )
    cudaMemcpy(array1_d, array1_h, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(array2_d, array2_h, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice);

    // allocating memory for resultant device array
    cudaMalloc((void **)&result_array_d, WIDTH * WIDTH * sizeof(int));
    cudaMalloc((void **)&M_result_array_d, WIDTH * WIDTH * sizeof(int));

    // calling kernal
    dim3 dimGrid(WIDTH / TILE_WIDTH, WIDTH / TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // Change if 0 to if 1 for running non shared code and make if 0 for shared memory code
    #if 0
                MatrixMul <<<dimGrid,dimBlock>>> ( array1_d , array2_d ,M_result_array_d , WIDTH);
    #endif
    #if 1
        // record events around kernel launch
        cudaEventRecord(event1, 0); // where 0 is the default stream
        MatrixMulSh<<<dimGrid, dimBlock>>>(array1_d, array2_d, M_result_array_d, WIDTH);
    #endif

    /* all gpu function blocked till kernel is working
     * copy back result_array_d to result_array_h */

    cudaMemcpy(M_result_array_h, M_result_array_d, WIDTH * WIDTH * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(event2, 0);

    // display the result array
    for (i = 0; i < WIDTH; i++)
    {
        for (j = 0; j < WIDTH; j++)
        {
            printf("%f ", M_result_array_h[i][j]);
        }
        printf("\n");
    }

    // synchronize
    cudaEventSynchronize(event1); // optional
    cudaEventSynchronize(event2); // wait for the event to be executed!
    
    // calculate time
    float dt_ms;
    
    cudaEventElapsedTime(&dt_ms, event1, event2);
    
    return 0;
}
