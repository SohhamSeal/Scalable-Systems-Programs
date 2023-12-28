/*
Write MPI parallel program to compute the area under a curve using trapezoidal rule.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

double func(double x)
{
    // Hardwired function
    return 4 / (1 + x * x);
}

double trappie(double a, double b, double n, double h)
{
    double sum = (func(a) + func(b)) / 2.0;
    for (int i = 1; i < n; i++)
        sum += func(a + i * h);
    sum *= h;
    return sum;
}

int main(int argc, char *argv[])
{
    int size, rank;
    double a, b, n, h;
    double ans;
    double local_a, local_b, local_n;
    double local_sum;
    
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 4)
    {
        if (rank == 0)
            printf("Enter the values of a,b and n in the cmd line exec statement!!\n");
        MPI_Finalize();
        exit(0);
    }
    
    a = atoi(argv[1]);
    b = atoi(argv[2]);
    n = atoi(argv[3]);
    h = (b - a) / n;
    local_n = n / size;
    local_a = a + rank * local_n * h;
    local_b = local_a + local_n * h;
    local_sum = trappie(local_a, local_b, local_n, h);
    
    // Reduce the local sums by adding them up
    MPI_Reduce(&local_sum, &ans, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Print the result in the root process
    if (rank == 0)
        printf("\nComputed integral value : %f\n", ans);
    
    MPI_Finalize();
    
    return 0;
}