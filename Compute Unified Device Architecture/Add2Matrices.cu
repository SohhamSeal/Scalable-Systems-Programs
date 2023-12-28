/*
Write CUDA program to add 2 Matrices A and B and store result in C.
*/

#include <stdio.h>
#include <stdlib.h>

#define N 2

__global__ void MatAdd(int A[][N], int B[][N], int C[][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    // create events
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    
    int A[N][N] = {{1, 2}, {3, 4}};
    int B[N][N] = {{5, 6}, {7, 8}};
    int C[N][N] = {{0, 0}, {0, 0}};
    
    int(*pA)[N], (*pB)[N], (*pC)[N];
    
    cudaMalloc((void **)&pA, (N * N) * sizeof(int));
    cudaMalloc((void **)&pB, (N * N) * sizeof(int));
    cudaMalloc((void **)&pC, (N * N) * sizeof(int));
    
    cudaMemcpy(pA, A, (N * N) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pB, B, (N * N) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pC, C, (N * N) * sizeof(int), cudaMemcpyHostToDevice);
    
    int numBlocks = 1;
    
    dim3 threadsPerBlock(N, N);
    
    // record events around kernel launch
    cudaEventRecord(event1, 0); // where 0 is the default stream
    
    MatAdd<<<numBlocks, threadsPerBlock>>>(pA, pB, pC);
    
    cudaMemcpy(C, pC, (N * N) * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(event2, 0);
    
    int i, j;
    
    printf("C = \n");
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
            printf("%d ", C[i][j]);
        printf("\n");
    }
    
    // synchronize
    cudaEventSynchronize(event1); // optional
    cudaEventSynchronize(event2); // wait for the event to be executed!
    
    // calculate time
    float dt_ms;
    
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("Time: %f", dt_ms);
    
    cudaFree(pA);
    cudaFree(pB);
    cudaFree(pC);
    
    return 0;
}