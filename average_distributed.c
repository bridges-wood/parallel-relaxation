#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <mpi.h>

enum log_level
{
    LOG_ALL = 0,
    LOG_DEBUG = 1,
    LOG_INFO = 2,
    LOG_WARN = 3,
    LOG_ERROR = 4,
    LOG_NONE = 5
};

// --- Begin function prototypes ---

/// @brief Print a square matrix to stdout
/// @param matrix The matrix to print
/// @param size The dimension of the matrix
void print_matrix(double *matrix, size_t size);

/// @brief Create a square matrix with 1s on the left and top sides and 0s
/// everywhere else.
/// @param size The dimension of the matrix
/// @param log_level The log level to use for debugging
/// @return A pointer to the matrix
double *matrix_init(size_t size, enum log_level log_level);

/// @brief Apply the relaxation technique to a matrix in parallel.
/// @param matrix The matrix to relax
/// @param size The dimension of the matrix
/// @param precision The precision to use for the relaxation, i.e. the maximum
/// difference between the average of a cell and its neighbours.
/// @param num_processes The number of processes to use
/// @param rank The rank of the current process
/// @param log_level The log level to use for debugging
/// @return A pointer to the relaxed matrix
double *relax_matrix_parallel(double *matrix, size_t size, double precision,
                              int num_processes, int rank,
                              enum log_level log_level);

/// @brief Use the relaxation technique to compute the average of a group of
/// cells in a matrix.
/// @param input The matrix to relax
/// @param result The matrix to store the result in
/// @param precision_reached A pointer to a boolean to store whether the
/// precision has been reached
/// @param dimension The dimension of the matrix
/// @param input_size The number of cells to relax
/// @param precision The precision to use for the relaxation, i.e. the maximum
/// difference between the average of a cell and its neighbours.
/// @note The input and result matrices must be of size dimension x dimension.
void relax_cells(double *input, double *result, bool *precision_reached,
                 size_t size, size_t input_size, double precision);

// --- End function prototypes ---

int main(int argc, char *argv[])
{
    int rank, num_processes, err;
    // Initialise MPI
    err = MPI_Init(&argc, &argv);
    err += MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    err += MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    if (err != MPI_SUCCESS)
    {
        fprintf(stderr, "Error initializing MPI. \n");
        return 1;
    }

    // Create variables to store command line arguments
    enum log_level log_level;
    size_t size;
    double precision;

    // Check for correct number of arguments
    if (argc < 3 || argc > 4)
    {
        fprintf(stderr,
                "Usage: %s <matrix size> <precision> [log level]",
                argv[0]);
        return 1;
    }

    if (argc == 4)
    {
        // Parse log level
        log_level = atoi(argv[3]);
        // Validate log level -
        if (log_level < LOG_ALL || log_level > LOG_NONE)
        {
            printf("Invalid log level. Must be between %d and %d \n",
                   LOG_ALL, LOG_NONE);
            return 1;
        }
    }
    else
    {
        // Default log level
        log_level = LOG_NONE;
    }

    // Parse size
    size = atoi(argv[1]);
    // Validate size
    if (size < 2 || size > 10e6)
    {
        fprintf(stderr, "Size must be greater than 1 and less than 10e6\n");
        return 1;
    }

    // Parse precision
    precision = atof(argv[2]);
    // Validate precision
    if (precision <= 0)
    {
        fprintf(stderr, "Precision must be greater than 0");
        return 1;
    }

    if (num_processes > (size - 2))
    {
        printf(
            "Process count is greater than the number of rows. "
            "Please reduce the number of processes.\n");
        return 1;
    }

    double *matrix = matrix_init(size, log_level);

    matrix = relax_matrix_parallel(matrix, size, precision, num_processes, rank,
                                   log_level);
    if (log_level <= LOG_INFO && rank == 0)
    {
        printf("Final matrix:\n");
        print_matrix(matrix, size);
    }

    // Finalise the MPI environment
    MPI_Finalize();

    // Free memory
    free(matrix);
    if (log_level <= LOG_ALL)
        printf("Freed matrix at %p \n", matrix);

    return 0;
}

double *matrix_init(size_t size, enum log_level log_level)
{
    if (log_level <= LOG_DEBUG)
        fprintf(stdout, "Initializing matrix of size %zu x %zu \n", size, size);

    // Allocate memory for the matrix
    double *result = calloc(size * size, sizeof(double *));
    if (log_level <= LOG_ALL)
        printf("Allocated matrix at %p \n", result);

    // Randomly initialize the matrix
    for (size_t i = 0; i < size * size; i++)
    {
        // Set the sides to 1
        /** This is done to improve the interpretation and reproducibility of
            timing results across different matrix sizes. This particular
            pattern was chosen because it is easy to see the effect of the
            relaxation method on the matrix and ... [TODO: finish this]
        */
        if (i % size == 0 || i < size || i >= size * (size - 1) ||
            i % size == size - 1)
        {
            result[i] = 1;
        }
        // All other cells are initialised at 0 by calloc
    }

    if (log_level <= LOG_DEBUG)
    {
        printf("Matrix initialized \n");
        print_matrix(result, size);
    }

    return result;
}

void print_matrix(double *matrix, size_t size)
{
    printf("Display %zu x %zu matrix \n", size, size);
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            printf("%f ", matrix[i * size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Use the relaxation method to relax a 2d array
double *relax_matrix_parallel(double *matrix, size_t size, double precision,
                              int num_processes, int rank,
                              enum log_level log_level)
{
    // Create arrays to store the scatter and gather counts and displacements
    int *scatter_displ = calloc(num_processes, sizeof(int));
    int *scatter_count = calloc(num_processes, sizeof(int));
    int *gather_count = calloc(num_processes, sizeof(int));
    int *gather_displ = calloc(num_processes, sizeof(int));

    // Calculate the number of rows to send to each process
    size_t rows_per_process = (size - 2) / num_processes;
    // Calculate the number of rows that will be left over
    size_t rows_remainder = (size - 2) % num_processes;

    size_t scatter_offset = 0;
    for (int i = 0; i < num_processes; i++)
    {
        // Starting position for each chunk
        scatter_displ[i] = scatter_offset;
        // Send each row its working rows and the rows above and below
        scatter_count[i] = (rows_per_process + 2) * size;

        /* If there are rows left over, give each process one until there are
        none left */
        if (rows_remainder > 0)
        {
            scatter_count[i] += size;
            rows_remainder--;
        }

        // Update the offset for the next process
        scatter_offset += scatter_count[i] - (2 * size);

        // Only receive the rows that are being worked on
        gather_count[i] = scatter_count[i] - (2 * size);
        gather_displ[i] = scatter_displ[i] + size;
    }

    // Allocate memory for the send and receive buffers
    double *send_buffer = calloc(scatter_count[rank], sizeof(double));
    double *recv_buffer = calloc(gather_count[rank], sizeof(double));

    if (log_level <= LOG_DEBUG)
        printf("Process %d: Scatter displ: %d, Scatter count: %d"
               "Gather displ: %d, Gather count: % d\n ",
               rank, scatter_displ[rank], scatter_count[rank],
               gather_displ[rank], gather_count[rank]);

    // Store the convergence information for each process
    bool local_precision = true;
    bool global_precision = false;
    int iterations = 0;

    // Loop until the matrix converges
    while (!global_precision)
    {
        //  Scatter from root process into all other processes
        MPI_Scatterv(matrix, scatter_count, scatter_displ, MPI_DOUBLE,
                     send_buffer, scatter_count[rank], MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        // Each process relaxes its own section of the matrix
        relax_cells(send_buffer, recv_buffer, &local_precision, size,
                    scatter_count[rank], precision);

        // Check convergence information from each process
        MPI_Allreduce(&local_precision, &global_precision, 1, MPI_C_BOOL,
                      MPI_LAND, MPI_COMM_WORLD);

        // Gather from all other processes into root process
        MPI_Gatherv(recv_buffer, gather_count[rank], MPI_DOUBLE, matrix,
                    gather_count, gather_displ, MPI_DOUBLE, 0, MPI_COMM_WORLD);


        iterations++;
        if (log_level <= LOG_DEBUG && rank == 0)
        {
            printf("Finished iteration %d \n", iterations);
            print_matrix(matrix, size);
        }
    }

    if (log_level <= LOG_DEBUG)
        printf("Freeing memory \n");

    // Free the memory before exiting the program
    free(scatter_displ);
    free(scatter_count);
    free(gather_displ);
    free(gather_count);
    free(send_buffer);
    free(recv_buffer);

    return matrix;
}

void relax_cells(double *input, double *result, bool *precision_reached,
                 size_t size, size_t input_size, double precision)
{
    // Offset for the first working row - skip the first row
    int start = size;
    // Offset for the last working row - skip the last row
    int end = input_size - size - 1;
    int result_index = 0;

    *precision_reached = true;
    for (int input_index = start; input_index <= end; input_index++)
    {
        // If the cell is on the edge, skip it
        if (input_index % size == 0 || (input_index + 1) % size == 0)
        {
            result[result_index] = input[input_index];
        }
        else
        {
            // Relax the cell
            double new_value = (input[input_index - size] +
                                input[input_index + size] +
                                input[input_index - 1] +
                                input[input_index + 1]) /
                               4;
            result[result_index] = new_value;

            // Check the precision on that cell. Short circuit if possible
            if (*precision_reached &&
                fabs(new_value - input[input_index]) > precision)
                *precision_reached = false;
        }

        result_index++;
    }
}