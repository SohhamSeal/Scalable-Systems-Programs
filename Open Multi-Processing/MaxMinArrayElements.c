/*
Write a parallel program using Open-MP to find the minimum and maximum element in a
given array of large size N.
*/

#include <stdlib.h>
#include <time.h>
#include <omp.h>

// export omp_set_num_threads = <no_of_threads>
// Run: ./a.out <size of array>

int main(int argc, char *argv[])
{
    int N, *arr, max_ans, min_ans;
    double timer;
    
    if (argc != 2)
    {
        printf("Please enter the array size as a cmd argument!!\n");
        exit(0);
    }
    
    // Storing the array size in N
    N = atoi(argv[1]);
    if (N <= 0)
    {
        printf("Please enter a proper array size!!\n");
        exit(0);
    }
    
    arr = (int *)malloc(sizeof(int) * N);
    srand(time(0));
    
    // Random array creation
    for (int i = 0; i < N; i++)
        arr[i] = (rand() * rand() % 12345) % 12345;
    
    timer = omp_get_wtime();
    
    max_ans = arr[0];
    min_ans = arr[0];
    #pragma omp parallel for reduction(max : max_ans) reduction(min : min_ans)
    for (int i = 0; i < N; i++)
    {
        if (max_ans < arr[i])
            max_ans = arr[i];
        if (min_ans > arr[i])
            min_ans = arr[i];
    }
    
    timer = omp_get_wtime() - timer;
    
    printf("\nGlobal Maximum value : %d\n", max_ans);
    printf("Global Minimum value : %d\n\n", min_ans);
    printf("Total time taken: %f seconds\n", timer);
    
    return 0;
}