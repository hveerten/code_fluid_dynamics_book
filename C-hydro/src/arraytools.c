#include "arraytools.h"

double find_minimum_2D(double **A, int N_x, int N_y, int offset_x, int offset_y)
{
  // this minimum finding algorithm can also be parallelized using openMP,
  // where we use a reduction technique to first find the minimum of subsets
  // of the whole array and then compare these minima
  double val_min = 1e300;


  /*
  // parallel stuff
  int max_threads = omp_get_max_threads(); // get maximum possible number of
    // cores. Note that if -fopenmp not used, this will still return the
    // number of cores available on the system
  double val_min_local[max_threads];
  int i_l;
  
  for (i_l = 0; i_l < max_threads; i_l++)
    val_min_local[i_l] = 1e300;
  
  #pragma omp parallel
  {
    int i_x, i_y;
    int my_ID = omp_get_thread_num();
    int ID_max = omp_get_num_threads(); // this will be equal to max_threads
      // normally, but equal to one if -fopenmp not used (and the pragma
      // command is ignored)
  
    int start, stop;
    int block_size = N_x / omp_get_num_threads();
    
    start = block_size * my_ID;
    stop = start + block_size;
    // special case: if number of threads does not fit an integer times into
    // the total resolution, the last core will take care of the remainder as
    // well.
    if (my_ID == ID_max - 1)
    {
      stop = N_x;
    }
    
    for (i_x = offset_x + start; i_x < offset_x + stop; i_x++)
    {
      for (i_y = offset_y; i_y < N_y + offset_y; i_y++)
      {
        if (A[i_x][i_y] < val_min_local[my_ID])
          val_min_local[my_ID] = A[i_x][i_y];
      }
    }
  }

  // reduce the local results to final result
  for (i_l = 0; i_l < max_threads; i_l++)
    if (val_min > val_min_local[i_l]) val_min = val_min_local[i_l];
  */
  
  
  // as it stands, this nested loop cannot be meaningfully parallelized!
  int i_x, i_y;
  for (i_x = offset_x; i_x < N_x + offset_x; i_x++)
  {
    for (i_y = offset_y; i_y < N_y + offset_y; i_y++)
    {
      if (A[i_x][i_y] < val_min) val_min = A[i_x][i_y];
    }
  }
  
  
  return val_min;
}

////////////////////////////////////////////////////////////////////////////////

double find_minimum_3D(double ***A, int N_x, int N_y, int N_z, int offset_x,
  int offset_y, int offset_z)
{
  // this minimum finding algorithm can also be parallelized using openMP,
  // where we use a reduction technique to first find the minimum of subsets
  // of the whole array and then compare these minima
  double val_min = 1e300;

  /*
  // parallel stuff
  int max_threads = omp_get_max_threads(); // get maximum possible number of
    // cores. Note that if -fopenmp not used, this will still return the
    // number of cores available on the system
  double val_min_local[max_threads];
  int i_l;
  
  for (i_l = 0; i_l < max_threads; i_l++)
    val_min_local[i_l] = 1e300;
  
  #pragma omp parallel
  {
    int i_x, i_y, i_z;
    int my_ID = omp_get_thread_num();
    int ID_max = omp_get_num_threads(); // this will be equal to max_threads
      // normally, but equal to one if -fopenmp not used (and the pragma
      // command is ignored)
  
    int start, stop;
    int block_size = N_x / omp_get_num_threads();
    
    start = block_size * my_ID;
    stop = start + block_size;
    // special case: if number of threads does not fit an integer times into
    // the total resolution, the last core will take care of the remainder as
    // well.
    if (my_ID == ID_max - 1)
    {
      stop = N_x;
    }
    
    for (i_x = offset_x + start; i_x < offset_x + stop; i_x++)
    {
      for (i_y = offset_y; i_y < N_y + offset_y; i_y++)
      {
        for (i_z = offset_z; i_z < N_z + offset_z; i_z++)
        {
          if (A[i_x][i_y][i_z] < val_min_local[my_ID])
            val_min_local[my_ID] = A[i_x][i_y][i_z];
        }
      }
    }
  }

  // reduce the local results to final result
  for (i_l = 0; i_l < max_threads; i_l++)
    if (val_min > val_min_local[i_l]) val_min = val_min_local[i_l];
  */
  
  
  // as it stands, this nested loop cannot be meaningfully parallelized!
  int i_x, i_y, i_z;
  for (i_x = offset_x; i_x < N_x + offset_x; i_x++)
  {
    for (i_y = offset_y; i_y < N_y + offset_y; i_y++)
    {
      for (i_z = offset_z; i_z < N_z + offset_z; i_z++)
      {
        if (A[i_x][i_y][i_z] < val_min) val_min = A[i_x][i_y][i_z];
      }
    }
  } 
  
  return val_min;
}

////////////////////////////////////////////////////////////////////////////////

double** allocate_2D_array_double(const int N_x, const int N_y)
{
  double **A;
  int i_x;

  A = malloc(N_x * sizeof(double*));
  A[0] = malloc(N_x * N_y * sizeof(double));
  for (i_x = 0; i_x < N_x; i_x++)
  {
    A[i_x] = A[0] + i_x * N_y;
  }
  
  return A;
}

////////////////////////////////////////////////////////////////////////////////

void free_2D_array_double(double **A)
{
  free(A[0]);
  free(A);
}

////////////////////////////////////////////////////////////////////////////////

double*** allocate_3D_array_double(const int N_x, const int N_y, const int N_z)
{
  double ***A;
  int i_x, i_y;

  A = malloc(N_x * sizeof(double**));

  A[0] = malloc(N_x * N_y * sizeof(double*));

  for(i_x = 0; i_x < N_x; i_x++)
    A[i_x] = A[0] + i_x * N_y;
   
  A[0][0] = malloc(N_x * N_y * N_z * sizeof(double));

  // this loop generalizes the pattern from the 2D array code. Again the 
  // i_x = i_y = 0 iteration is redundant.
  for(i_x = 0; i_x < N_x; i_x++)
    for (i_y = 0; i_y < N_y; i_y++)
    {
      A[i_x][i_y] = A[0][0] + i_x * N_y * N_z + i_y * N_z;
    }

  return A;
}

////////////////////////////////////////////////////////////////////////////////

void free_3D_array_double(double ***A)
{
  free(A[0][0]);
  free(A[0]);
  free(A);
}

////////////////////////////////////////////////////////////////////////////////

char*** allocate_3D_array_char(const int N_x, const int N_y, const int N_z)
{
  char ***A;
  int i_x, i_y;

  A = malloc(N_x * sizeof(char**));

  A[0] = malloc(N_x * N_y * sizeof(char*));

  for(i_x = 0; i_x < N_x; i_x++)
    A[i_x] = A[0] + i_x * N_y;
   
  A[0][0] = malloc(N_x * N_y * N_z * sizeof(char));

  // this loop generalizes the pattern from the 2D array code. Again the 
  // i_x = i_y = 0 iteration is redundant.
  for(i_x = 0; i_x < N_x; i_x++)
    for (i_y = 0; i_y < N_y; i_y++)
    {
      A[i_x][i_y] = A[0][0] + i_x * N_y * N_z + i_y * N_z;
    }

  return A;
}

////////////////////////////////////////////////////////////////////////////////

void free_3D_array_char(char ***A)
{
  free(A[0][0]);
  free(A[0]);
  free(A);
}

////////////////////////////////////////////////////////////////////////////////

double**** allocate_4D_array_double(const int N_w, const int N_x, const int N_y, const int N_z)
{
  double ****A;
  int i_w, i_x, i_y;

  A = malloc(N_w * sizeof(double***));
  A[0] = malloc(N_w * N_x * sizeof(double**));
  
  for (i_w = 0; i_w < N_w; i_w++)
    A[i_w] = A[0] + i_w * N_x;
  
  A[0][0] = malloc(N_w * N_x * N_y * sizeof(double*));
  
  for(i_w = 0; i_w < N_w; i_w++)
    for (i_x = 0; i_x < N_x; i_x++)
    {
      A[i_w][i_x] = A[0][0] + i_w * N_x * N_y + i_x * N_y;
    }

  A[0][0][0] = malloc(N_w * N_x * N_y * N_z * sizeof(double));
  
  for(i_w = 0; i_w < N_w; i_w++)
    for (i_x = 0; i_x < N_x; i_x++)
      for (i_y = 0; i_y < N_y; i_y++)
      {
        A[i_w][i_x][i_y] = A[0][0][0] + i_w * N_x * N_y * N_z + i_x * N_y * N_z
          + i_y * N_z;
      }

  return A;
}

////////////////////////////////////////////////////////////////////////////////

double***** allocate_5D_array_double(const int N_v, const int N_w,
  const int N_x, const int N_y, const int N_z)
{
  double *****A;
  int i_v, i_w, i_x, i_y;

  A = malloc(N_v * sizeof(double****));
  A[0] = malloc(N_v * N_w * sizeof(double***));
  
  for (i_v = 0; i_v < N_v; i_v++)
    A[i_v] = A[0] + i_v * N_w;
  
  A[0][0] = malloc(N_v * N_w * N_x * sizeof(double**));
  
  for(i_v = 0; i_v < N_v; i_v++)
    for (i_w = 0; i_w < N_w; i_w++)
    {
      A[i_v][i_w] = A[0][0] + i_v * N_w * N_x + i_w * N_x;
    }

  A[0][0][0] = malloc(N_v * N_w * N_x * N_y * sizeof(double*));
  
  for(i_v = 0; i_v < N_v; i_v++)
    for (i_w = 0; i_w < N_w; i_w++)
      for (i_x = 0; i_x < N_x; i_x++)
      {
        A[i_v][i_w][i_x] = A[0][0][0] + i_v * N_w * N_x * N_y + i_w * N_x * N_y
          + i_x * N_y;
      }

  A[0][0][0][0] = malloc(N_v * N_w * N_x * N_y * N_z *sizeof(double));
  for(i_v = 0; i_v < N_v; i_v++)
    for (i_w = 0; i_w < N_w; i_w++)
      for (i_x = 0; i_x < N_x; i_x++)
        for (i_y = 0; i_y < N_y; i_y++)
        {
          A[i_v][i_w][i_x][i_y] = A[0][0][0][0] + 
            i_v * N_w * N_x * N_y * N_z + 
            i_w * N_x * N_y * N_z +
            i_x * N_y * N_z +
            i_y * N_z;
        }

  return A;
}

////////////////////////////////////////////////////////////////////////////////
void free_4D_array_double(double ****A)
{
  free(A[0][0][0]);
  free(A[0][0]);
  free(A[0]);
  free(A);
}

////////////////////////////////////////////////////////////////////////////////

void free_5D_array_double(double *****A)
{
  free(A[0][0][0][0]);
  free(A[0][0][0]);
  free(A[0][0]);
  free(A[0]);
  free(A);
}

////////////////////////////////////////////////////////////////////////////////
