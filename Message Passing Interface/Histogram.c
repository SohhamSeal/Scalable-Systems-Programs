/*
Write MPI program to generate a histogram of Marks scores in a class of strength 125. Choose the
appropriate bin size for the problem.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define ARRAY_SIZE 125

int main(int argc, char *argv[])
{
    int size, rank;
    int bin_count[10] = {0}, total_count[10] = {0}, *local_array;
    
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Initialize the array in the root process
    if (rank == 0)
    {
        local_array = (int *)malloc(sizeof(int) * ARRAY_SIZE);
        srand(time(0));
        for (int i = 0; i < ARRAY_SIZE; i++)
            local_array[i] = (rand() + rand() % 123456) % 100;
    }
    else
        local_array = (int *)malloc(sizeof(int) * ARRAY_SIZE / size);
    
    // Scatter the array to all the processes
    MPI_Scatter(local_array, ARRAY_SIZE / size, MPI_INT, local_array, ARRAY_SIZE / size, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Update bin frequencies for each local array
    for (int i = 0; i < ARRAY_SIZE / size; i++)
        if (local_array[i] && local_array[i] % 10 == 0)
            bin_count[local_array[i] / 10 - 1]++;
        else
            bin_count[local_array[i] / 10]++;
    
    // Reduce to find the total sum of frequencies in the root process
    MPI_Reduce(bin_count, total_count, 10, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Print result in the root process
    if (rank == 0)
    {
        printf("\nHistogram of marks for 125 students::\n");
        for (int i = 0; i < 10; i++)
        {
            printf("%2d - %3d : ", i * 10, i * 10 + 10);
            for (int j = 1; j < total_count[i]; j++)
                printf("■");
            printf(" (%d)\n", total_count[i]);
        }
        printf("\n");
    }
    
    free(local_array);
    
    MPI_Finalize();
    
    return 0;
}