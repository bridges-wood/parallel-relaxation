#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
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
  int num_threads;
  enum log_level log_level;
  double **matrix;
} shared_args;

typedef struct
{
  int id;
  int start_i;
  int start_j;
  int cells;
  double **original_matrix;
  double **new_matrix;
} thread_args;

/* True if the required precision has been reached on all threads, false
otherwise.*/
bool PRECISION_REACHED = false;
/* An array of flags, one for each thread. True if the required precision has
been reached on the thread, false otherwise. */
bool *THREAD_PRECISION_REACHED;
// Barrier for all threads.
pthread_barrier_t barrier;

// --- Begin function prototypes ---

// Print a square matrix
void print_matrix(double **matrix);

/* Create an n x n matrix with 1s on the left and top sides and 0s everywhere
else. */
double **matrix_init();

// Apply the relaxation technique to a matrix in parallel.
double **relax_matrix_parallel(double **matrix);

/* Use the relaxation technique to compute the average of a group of cells in
a matrix. */
void *relax_cells(void *args);

// Determine the arguments for all child threads.
void determine_thread_data(
    double **original_matrix,
    double **new_matrix,
    thread_args *thread_data);

// --- End function prototypes ---

// Program Entry
int main(int argc, char *argv[])
{
  // Check for correct number of arguments
  if (argc < 4 || argc > 5)
  {
    fprintf(stderr, "Usage: %s <matrix size> <precision> <num threads> [log level]", argv[0]);
    return 1;
  }

  if (argc == 5)
  {
    shared_args.log_level = atoi(argv[4]);
    if (shared_args.log_level < LOG_ALL || shared_args.log_level > LOG_NONE)
    {
      printf("Invalid log level. Must be between %d and %d \n", LOG_ALL, LOG_NONE);
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

  // Parse num_threads
  shared_args.num_threads = atoi(argv[3]);
  // Validate num_threads
  if (shared_args.num_threads < 1)
  {
    fprintf(stderr, "Thread count must be greater than 0");
    return 1;
  }
  else if (shared_args.num_threads >
           (shared_args.size - 2) * (shared_args.size - 2))
  {
    printf(
        "Thread count is greater than the number of cells. "
        "Using %zu threads.\n",
        (shared_args.size - 2) * (shared_args.size - 2));
    shared_args.num_threads = (shared_args.size - 2) * (shared_args.size - 2);
  }

  // Initialise THREAD_PRECISION_REACHED
  THREAD_PRECISION_REACHED = calloc(shared_args.num_threads, sizeof(bool));
  for (int i = 0; i < shared_args.num_threads; i++)
    THREAD_PRECISION_REACHED[i] = true;

  if (shared_args.log_level <= LOG_ALL)
    printf("Allocated thread precision flags at %p \n", THREAD_PRECISION_REACHED);

  double **a = matrix_init();

  a = relax_matrix_parallel(a);
  if (shared_args.log_level <= LOG_INFO)
  {
    print_matrix(a);
  }

  for (int i = 0; i < shared_args.size; i++)
  {
    free(a[i]);
    if (shared_args.log_level <= LOG_ALL)
      printf("Freed row %d at %p \n", i, a[i]);
  }

  free(a);
  if (shared_args.log_level <= LOG_ALL)
    printf("Freed matrix at %p \n", a);

  free(THREAD_PRECISION_REACHED);
  if (shared_args.log_level <= LOG_ALL)
    printf("Freed thread precision flags at %p \n", THREAD_PRECISION_REACHED);

  return 0;
}

double **matrix_init()
{
  // Initialise random number generator
  srand(42);

  if (shared_args.log_level <= LOG_DEBUG)
    printf("Initializing matrix of size %zu x %zu \n",
           shared_args.size, shared_args.size);

  // Allocate memory for the matrix
  double **result = calloc(shared_args.size, sizeof(double *));
  if (shared_args.log_level <= LOG_ALL)
    printf("Allocated matrix at %p \n", result);
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

    if (shared_args.log_level <= LOG_ALL)
      printf("Allocated row %zu at %p \n", i, result[i]);

    // Randomly initialise the matrix
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

double **relax_matrix_parallel(double **matrix)
{
  // Copy the original matrix to a new matrix.
  double **new_matrix = matrix_init();

  thread_args *thread_data = calloc(
      shared_args.num_threads,
      sizeof(thread_args));
  if (shared_args.log_level <= LOG_ALL)
    printf("Allocated thread data at %p \n", thread_data);

  pthread_t *threads = calloc(shared_args.num_threads, sizeof(pthread_t));
  if (shared_args.log_level <= LOG_ALL)
    printf("Allocated threads at %p \n", threads);

  if (shared_args.log_level <= LOG_DEBUG)
    printf("Allocated memory for new matrix and thread data \n");

  determine_thread_data(matrix, new_matrix, thread_data);
  // +1 for controlling the main thread.
  pthread_barrier_init(
      &barrier,
      NULL,
      shared_args.num_threads + 1);

  // Create threads
  for (int i = 0; i < shared_args.num_threads; i++)
    pthread_create(
        &threads[i],
        NULL,
        relax_cells,
        &thread_data[i]);

  if (shared_args.log_level <= LOG_INFO)
    printf("Threads created \n");

  int iterations = 0;
  while (true)
  {
    // Wait for all threads to finish.
    pthread_barrier_wait(&barrier);
    for (int i = 0; i < shared_args.num_threads; i++)
    {
      if (THREAD_PRECISION_REACHED[i] == false)
      {
        PRECISION_REACHED = false;
        break;
      }
      else
      {
        PRECISION_REACHED = true;
      }
    }

    // Check if precision has been reached.
    if (PRECISION_REACHED)
    {
      break;
    }
    else
    {
      // If not, reset precision array and sync threads.
      for (int i = 0; i < shared_args.num_threads; i++)
      {
        THREAD_PRECISION_REACHED[i] = true;
      }
    }

    iterations++;
    if (shared_args.log_level <= LOG_INFO)
      printf("Finished iteration %d \n", iterations);

    pthread_barrier_wait(&barrier);
  }

  pthread_barrier_wait(&barrier);

  // Join threads
  for (int i = 0; i < shared_args.num_threads; i++)
  {
    if (shared_args.log_level <= LOG_DEBUG)
      printf("Joining thread %d \n", i);
    pthread_join(threads[i], NULL);
  }

  if (shared_args.log_level <= LOG_DEBUG)
    printf("Threads joined, freeing memory \n");

  // Free the memory before exiting the program
  pthread_barrier_destroy(&barrier);

  free(thread_data);
  if (shared_args.log_level <= LOG_ALL)
    printf("Freed thread data at %p \n", thread_data);

  free(threads);
  if (shared_args.log_level <= LOG_ALL)
    printf("Freed threads at %p \n", threads);

  for (int i = 0; i < shared_args.size; i++)
  {
    free(new_matrix[i]);
    if (shared_args.log_level <= LOG_ALL)
      printf("Freed row %d at %p \n", i, new_matrix[i]);
  }
  free(new_matrix);
  if (shared_args.log_level <= LOG_ALL)
    printf("Freed matrix at %p \n", new_matrix);

  return matrix;
}

void *relax_cells(void *args)
{
  thread_args *t_args = (thread_args *)args;

  // While the required precision has not been reached
  while (true)
  {
    int i = t_args->start_i;
    int j = t_args->start_j;

    if (shared_args.log_level <= LOG_DEBUG)
      printf("Thread %d starting at (%d, %d) \n", t_args->id, i, j);
    for (int c = 0; c < t_args->cells; c++)
    {
      // Compute the average of the surrounding cells
      double new_value = (t_args->original_matrix[i][j - 1] +
                          t_args->original_matrix[i][j + 1] +
                          t_args->original_matrix[i - 1][j] +
                          t_args->original_matrix[i + 1][j]) /
                         4.0;
      // Update the new matrix
      t_args->new_matrix[i][j] = new_value;

      // Check if the precision has been reached.
      double difference = fabs(new_value - t_args->original_matrix[i][j]);
      if (difference > shared_args.precision)
      {
        // If changes fall outside of precision update global flag.
        THREAD_PRECISION_REACHED[t_args->id] = false;
      }

      j++;
      // If we have reached the end of the row, move to the next row.
      if (j == shared_args.size - 1)
      {
        j = 1;
        i++;
      }
    }

    if (shared_args.log_level <= LOG_DEBUG)
      printf("Thread %d finished iteration \n", t_args->id);

    // Wait for all computation to finish.
    pthread_barrier_wait(&barrier);
    // Wait for main thread to check precision.
    pthread_barrier_wait(&barrier);

    if (PRECISION_REACHED)
      break;

    // Swap the matrices.
    double **temp = t_args->original_matrix;
    t_args->original_matrix = t_args->new_matrix;
    t_args->new_matrix = temp;
  }

  if (shared_args.log_level <= LOG_INFO)
    printf("Thread %d finished \n", t_args->id);
  // Terminate self
  return NULL;
}

void determine_thread_data(
    double **original_matrix,
    double **new_matrix,
    thread_args *thread_data)
{
  // ! An int may be too small
  uint64_t inner_cells = (shared_args.size - 2) * (shared_args.size - 2);
  if (shared_args.log_level <= LOG_DEBUG)
    printf("Total inner cells: %zu \n", inner_cells);
  int cells_per_thread = inner_cells / shared_args.num_threads;
  int remainder = inner_cells % cells_per_thread;
  int start_i = 1;
  int start_j = 1;
  for (int i = 0; i < shared_args.num_threads; i++)
  {
    thread_data[i].id = i;
    thread_data[i].start_i = start_i;
    thread_data[i].start_j = start_j;
    thread_data[i].cells = cells_per_thread;
    if (remainder > 0)
    {
      thread_data[i].cells++;
      remainder--;
    }

    thread_data[i].original_matrix = original_matrix;
    thread_data[i].new_matrix = new_matrix;

    if (shared_args.log_level <= LOG_DEBUG)
      printf("Thread %d will compute %d cells starting at (%d, %d) \n", i,
             thread_data[i].cells, start_i, start_j);

    int cells_remaining = thread_data[i].cells;
    while (cells_remaining > 0)
    {
      start_j++;
      cells_remaining--;
      if (start_j == shared_args.size - 1)
      {
        start_j = 1;
        start_i++;
      }
    }
  }
}