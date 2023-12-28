/*
Write MPI parallel program to compute the area under a curve using trapezoidal rule.
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double func(double x)
{
    // Hardwired function
    return 4.0 / (1 + x * x);
}

int main(int argc, char *argv[])
{
    int N;
    double a, b, ans;
    
    if (argc != 4)
    {
        printf("Please enter proper number of values in the cmd: (a,b,n)!!\n");
        exit(0);
    }
    
    a = atoi(argv[1]);
    b = atoi(argv[2]);
    N = atoi(argv[3]);
    if (N <= 0)
    {
        printf("Wrong input for number of steps!!\n");
        exit(0);
    }
    
    ans = (func(0) + func(1)) / 2.0;
    #pragma omp parallel for reduction(+ : ans)
    for (int i = 1; i < N; i++)
        ans += func(a + i * ((b - a) * 1.0 / N));
    
    printf("\nComputed integral value : %f\n", ans * (b - a) / N);
    
    return 0;
}