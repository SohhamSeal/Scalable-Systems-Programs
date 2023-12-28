/*
Write an MPI program to find maximum value in an array of 600 integers with 6 processes and
print the result in root process using MPI_Reduce call. Compute time taken by the program using
MPI_Wtime() function.
*/


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define ARRAY_SIZE 600

int main(int argc, char *argv[])
{
    int size, rank;
    int *local_array;
    int global_max, local_max;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 6)
    {
        if (rank == 0)
            printf("Please try to run the program with 6 processes!!\n");
        MPI_Finalize();
        exit(0);
    }
    
    // Initialize the array in the root process
    if (rank == 0)
    {
        local_array = (int *)malloc(sizeof(int) * ARRAY_SIZE);
        srand(time(0));
        for (int i = 0; i < ARRAY_SIZE; i++)
            local_array[i] = rand() % 12345;
    }
    else
        local_array = (int *)malloc(sizeof(int) * (ARRAY_SIZE / 6));
    
    // Scatter the array to all the processes
    MPI_Scatter(local_array, ARRAY_SIZE / 6, MPI_INT, local_array, ARRAY_SIZE / 6, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Start measuring time
    start_time = MPI_Wtime();
    
    // Find local maximum
    local_max = local_array[0];
    for (int i = 0; i < ARRAY_SIZE / 6; i++)
        if (local_array[i] > local_max)
            local_max = local_array[i];
    
    // Reduce to find the global maximum in the root process
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Stop measuring time
    end_time = MPI_Wtime();
    
    // Print result in the root process
    if (rank == 0)
    {
        printf("Global Maximum value: %d\n", global_max);
        printf("Total time taken: %f seconds\n", end_time - start_time);
    }
    
    free(local_array);
    
    MPI_Finalize();
    
    return 0;
}