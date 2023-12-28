/*
The value of π is computed mathematically as follows:
        integral (4/1+x^2) from 0 to 1
Write an Open-MP program to compute π. Compare execution time for serial code and parallel code.
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double func(double x)
{
    return 4.0 / (1 + x * x);
}

int main(int argc, char *argv[])
{
    int N;
    double pi, timer;
    
    if (argc != 2)
    {
        printf("Please enter the number of intervals in the cmd!!\n");
        exit(0);
    }
    
    N = atoi(argv[1]);
    
    timer = omp_get_wtime();
    
    pi = (func(0) + func(1)) / 2.0;
    #pragma omp parallel for reduction(+ : pi)
    for (int i = 1; i < N; i++)
        pi += func(i * (1.0 / N));
    
    timer = omp_get_wtime() - timer;
    
    printf("\nComputed Pi value : %f\n", pi / N);
    printf("Total time taken: %f seconds\n", timer);
    
    return 0;
}