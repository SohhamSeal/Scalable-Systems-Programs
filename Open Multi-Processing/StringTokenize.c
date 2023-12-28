/*
Write Open-MP parallel program to tokenize a given text.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define MAX_LINES 3
#define MAX_LINE 30

void Get_text(char *lines[], int *line_count_p); // save the sentences in seperate streams
void Tokenize(char *lines[], int line_count, int thread_count);

int main(int argc, char *argv[])
{
    int thread_count, i;
    char *lines[100];
    int line_count = 3;
    
    // Check for error in cmd!!
    if (argc != 2)
    {
        printf("Please enter proper number of threads in cmd!!\n");
        exit(0);
    }
    
    thread_count = atoi(argv[1]);
    
    // Accept text from the user
    printf("Enter text:\n");
    
    Get_text(lines, &line_count);
    
    printf("\n");
    
    // Tokenize and display
    Tokenize(lines, line_count, thread_count);
    for (i = 0; i < line_count; i++)
        if (lines[i] != NULL)
            free(lines[i]);
    
    return 0;
}

void Get_text(char *lines[], int *line_count)
{
    char *line = malloc(MAX_LINE * sizeof(char));
    int i;
    char *fg_rv;
    
    for (i = 0; i < 4; i++)
    {
        fg_rv = fgets(line, MAX_LINE, stdin);
        line = malloc(MAX_LINE * sizeof(char));
        lines[i] = line;
    }
    
    *line_count = i;
}

void Tokenize(char *lines[] /* in/out */, int line_count /* in */, int thread_count /* in */)
{
    int my_rank, i, j;
    char *my_token, *saveptr;

    #pragma omp parallel num_threads(thread_count) default(none) private(my_rank, i, j, my_token, saveptr) shared(lines, line_count)
    {
        my_rank = omp_get_thread_num();
        #pragma omp for
        for (i = 0; i < line_count; i++)
        {
            printf("\nThread %d > line %d = %s", my_rank, i, lines[i]);
            j = 0;
            my_token = strtok_r(lines[i], " \t\n", &saveptr);
            while (my_token != NULL)
            {
                printf("Thread %d > token %d = %s\n", my_rank, j++, my_token);
                my_token = strtok_r(NULL, " \t\n", &saveptr);
            }
            if (lines[i] != NULL)
                printf("Thread %d > After tokenizing, my line= %s\n", my_rank, lines[i]);
        }
    }
}