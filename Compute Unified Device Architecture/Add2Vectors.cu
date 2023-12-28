/*
Write CUDA program to add 2 vectors A and B and store result in C.
*/

#include <stdio.h>

const short N = 6;

// CUDA Kernel for Vector Addition
__global__ void Vector_Addition(const int *dev_a, const int *dev_b, int *dev_c)
{
    // Get the id of thread within a block
    unsigned short tid = blockIdx.x;
    if (tid < N) // check the boundary condition for the threads
        dev_c[tid] = dev_a[tid] + dev_b[tid];
}

int main(void)
{
    // create events
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    
    // Host array
    int Host_a[N], Host_b[N], Host_c[N];
    
    // Device array
    int *dev_a, *dev_b, *dev_c;
    
    // Allocate the memory on the GPU
    cudaMalloc((void **)&dev_a, N * sizeof(int));
    cudaMalloc((void **)&dev_b, N * sizeof(int));
    cudaMalloc((void **)&dev_c, N * sizeof(int));
    
    // fill the Host array with random elements on the CPU
    for (int i = 0; i < N; i++)
    {
        Host_a[i] = i + i;
        Host_b[i] = i * i;
    }
    
    cudaEventRecord(event1, 0);
    
    // Copy Host array to Device array
    cudaMemcpy(dev_a, Host_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, Host_b, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Make a call to GPU kernel
    Vector_Addition<<<N, 1>>>(dev_a, dev_b, dev_c);
    
    // Copy back to Host array from Device array
    cudaMemcpy(Host_c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(event2, 0);
    
    // synchronize
    cudaEventSynchronize(event1); // optional
    cudaEventSynchronize(event2); // wait for the event to be executed!
    
    // calculate time
    float dt_ms;
    
    cudaEventElapsedTime(&dt_ms, event1, event2);
    
    for (int i = 0; i < N; i++)
        printf("%d + %d = %d\n", Host_a[i], Host_b[i], Host_c[i]);
    
    printf("Time: %f\n", dt_ms);
    
    // Free the Device array memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    return 0;
}