/*
Write a MPI parallel program to find the minimum and maximum element in a given array of
large size N. The code should allow easy changing of parameters like the number of processors, the input
size n. Use random function to generate random numbers.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char *argv[])
{
    int size, rank;
    int *local_array, N;
    int global_max, global_min, local_min, local_max;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Check for array size parameter's existance
    if (argc != 2)
    {
        if (rank == 0)
            printf("Please enter the array size as a cmd argument!!\n");
        MPI_Finalize();
        exit(0);
    }
    
    // Store the array size in N
    N = atoi(argv[1]);
    
    // Initialize the array in the root process
    if (rank == 0)
    {
        local_array = (int *)malloc(sizeof(int) * N);
        srand(time(0));
        for (int i = 0; i < N; i++)
            local_array[i] = (rand() + rand() % 123456) % 12345;
    }
    else
        local_array = (int *)malloc(sizeof(int) * (N / size));
    
    // Scatter the array to all the processes
    MPI_Scatter(local_array, N / size, MPI_INT, local_array, N / size, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Start measuring time
    start_time = MPI_Wtime();
    
    // Find local maximum
    local_max = local_array[0];
    for (int i = 0; i < N / size; i++)
    {
        if (local_min > local_array[i])
            local_min = local_array[i];
        if (local_array[i] > local_max)
            local_max = local_array[i];
    }
    
    // Reduce to find the global maximum in the root process
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    
    // Stop measuring time
    end_time = MPI_Wtime();
    
    // Print result in the root process
    if (rank == 0)
    {
        printf("\nGlobal Maximum value: %d\n", global_max);
        printf("Global Minimum value: %d\n\n", global_min);
        printf("Total time taken: %f seconds\n", end_time - start_time);
    }
    
    free(local_array);
    
    MPI_Finalize();
    
    return 0;
}