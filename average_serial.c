#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

enum log_level
{
    LOG_ALL = 0,
    LOG_DEBUG = 1,
    LOG_INFO = 2,
    LOG_WARN = 3,
    LOG_ERROR = 4,
    LOG_NONE = 5
};

// Global variables
struct
{
    size_t size;
    double precision;
    enum log_level log_level;
    double **matrix;
} shared_args;

// Print a square matrix
void print_matrix(double **matrix)
{
    printf("Display %zu x %zu matrix \n", shared_args.size, shared_args.size);
    for (int i = 0; i < shared_args.size; i++)
    {
        for (int j = 0; j < shared_args.size; j++)
        {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Create an n x n matrix with 1s on the left and top sides and 0s everywhere else.
double **matrix_init()
{
    // Initialise random number generator
    srand(42);

    if (shared_args.log_level <= LOG_DEBUG)
        printf("Initializing matrix of size %zu x %zu \n",
               shared_args.size, shared_args.size);

    // Allocate memory for the matrix
    double **result = calloc(shared_args.size, sizeof(double *));
    if (shared_args.matrix == NULL)
    {
        shared_args.matrix = malloc(shared_args.size * sizeof(double *));
    }
    else
    {
        for (int i = 0; i < shared_args.size; i++)
        {
            result[i] = calloc(shared_args.size, sizeof(double));
            for (int j = 0; j < shared_args.size; j++)
            {
                result[i][j] = shared_args.matrix[i][j];
            }
        }
        return result;
    }

    for (size_t i = 0; i < shared_args.size; i++)
    {
        // Allocate memory for each row
        result[i] = calloc(shared_args.size, sizeof(double));
        if (shared_args.matrix[i] == NULL)
        {
            shared_args.matrix[i] = malloc(shared_args.size * sizeof(double));
        }

        // Randomly initialize the matrix
        for (size_t j = 0; j < shared_args.size; j++)
        {
            if (i == 0 || j == 0)
            {
                result[i][j] = shared_args.matrix[i][j] = 1;
            }
            else
            {
                result[i][j] = shared_args.matrix[i][j] = (double)rand() / (double)RAND_MAX;
            }
        }
    }

    if (shared_args.log_level <= LOG_DEBUG)
    {
        printf("Matrix initialized \n");
        print_matrix(result);
    }

    return result;
}

// Use the relaxation technique to compute the average of a 2d array to a given precision
double **serial_average_matrix(double **matrix)
{
    double **new_matrix = matrix_init();
    bool still_changing = true;
    int iteration = 0;
    while (still_changing)
    {
        still_changing = false;
        // Avoid boundary values - they're fixed
        for (int i = 1; i < shared_args.size - 2; i++)
        {
            for (int j = 1; j < shared_args.size - 2; j++)
            {
                // Compute average using the relaxation approach
                double new_val = 0.25 * (matrix[i - 1][j] + matrix[i + 1][j] + matrix[i][j - 1] + matrix[i][j + 1]);
                double deviation = fabs(matrix[i][j] - new_val);
                if (deviation > shared_args.precision)
                {
                    still_changing = true;
                }
                new_matrix[i][j] = new_val;
            }
        }

        // Swap the matrices
        double **temp = matrix;
        matrix = new_matrix;
        new_matrix = temp;

        if (shared_args.log_level <= LOG_DEBUG)
        {
            printf("Matrix after iteration %d\n", iteration);
            print_matrix(matrix);
        }

        iteration++;
    }

    // Free the new matrix
    for (int i = 0; i < shared_args.size; i++)
    {
        free(new_matrix[i]);
    }
    free(new_matrix);

    return matrix;
}

// Program Entry
int main(int argc, char *argv[])
{
    // Check for correct number of arguments
    if (argc < 3 || argc > 4)
    {
        fprintf(stderr, "Usage: %s <size> <precision> [log_level] \n", argv[0]);
        return 1;
    }

    if (argc == 4)
    {
        shared_args.log_level = atoi(argv[3]);
        if (shared_args.log_level < LOG_ALL || shared_args.log_level > LOG_NONE)
        {
            fprintf(stderr, "Invalid log level. Must be between %d and %d \n", LOG_ALL, LOG_NONE);
            return 1;
        }
    }
    else
    {
        shared_args.log_level = LOG_NONE;
    }

    // Parse size
    shared_args.size = atoi(argv[1]);
    // Validate size
    if (shared_args.size < 2 || shared_args.size > 10e6)
    {
        fprintf(stderr, "Size must be greater than 1 and less than 10e6\n");
        return 1;
    }

    // Parse precision
    shared_args.precision = atof(argv[2]);
    // Validate precision
    if (shared_args.precision <= 0)
    {
        fprintf(stderr, "Precision must be greater than 0");
        return 1;
    }

    double **a = matrix_init();
    a = serial_average_matrix(a);
    if (shared_args.log_level <= LOG_INFO)
    {
        print_matrix(a);
    }

    // Free the memory before exiting the program
    for (size_t i = 0; i < shared_args.size; i++)
    {
        free(a[i]);
        free(shared_args.matrix[i]);
    }

    free(a);
    free(shared_args.matrix);
}
