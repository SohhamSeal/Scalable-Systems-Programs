/*
Write an OpenMP parallel program to multiply a matrix with a vector.
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int main(int argc, char *argv[])
{
    int n, m;
    int **mat, *vec, *ans;
    
    if (argc != 3)
    {
        printf("Please enter proper number of values in the cmd: (m,n)!!\n");
        exit(0);
    }
    
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    if (n <= 0 || m <= 0)
    {
        printf("Wrong input for dimensions!!\n");
        exit(0);
    }
    
    mat = (int **)malloc(sizeof(int *) * m);
    for (int i = 0; i < m; i++)
        mat[i] = (int *)malloc(sizeof(int) * n);
    vec = (int *)malloc(sizeof(int) * n);
    ans = (int *)malloc(sizeof(int) * m);
    
    srand(time(0));
    
    // Initialize random values to the matrix and vector
    #pragma omp parallel for
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
            mat[i][j] = (rand() + rand() % 123) % 123;
        vec[j] = (rand() + rand() % 123) % 123;
    }
    
    // Display
    printf("Matrix::\n");
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            printf("%d\t", mat[i][j]);
        printf("\n");
    }
    
    printf("\nVector::\n");
    for (int i = 0; i < m; i++)
        printf("%d\t", vec[i]);
    printf("\n");
    
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        ans[i] = 0;
        for (int j = 0; j < n; j++)
            ans[i] += mat[i][j] * vec[j];
    }
    
    printf("\n\nResultant vector ::\n");
    for (int i = 0; i < m; i++)
        printf("%d ", ans[i]);
    
    printf("\n\n");
    
    free(mat);
    free(vec);
    free(ans);
    
    return 0;
}