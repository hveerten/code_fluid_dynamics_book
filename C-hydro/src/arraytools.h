// A set of extra functions to work with multi-dimensional arrays as single
// blocks in memory

#ifndef ARRAYTOOLS_
#define ARRAYTOOLS_

#include <stdlib.h>
#include <omp.h>
#include "settings.h"

double find_minimum_2D(double **A, int N_x, int N_y, int offset_x, int offset_y);
// returns the minimum value of an array
// Arguments:
//   A         two-dimensional array
//   N_x       number of entries in x direction to check
//   N_y       number of entries in y direction to check
//   offset_x  starting entry in the x direction (eg for skipping ghosts)
//   offset_y  starting entry in the y direction (eg for skipping ghosts)
// Return:
//   minimum value in array

double find_minimum_3D(double ***A, int N_x, int N_y, int N_z, int offset_x,
  int offset_y, int offset_z);
// returns the minimum value of an array
// Arguments:
//   A         two-dimensional array
//   N_x       number of entries in x direction to check
//   N_x       number of entries in x direction to check
//   N_z       number of entries in z direction to check
//   offset_x  starting entry in the x direction (eg for skipping ghosts)
//   offset_y  starting entry in the y direction (eg for skipping ghosts)
//   offset_z  starting entry in the z direction (eg for skipping ghosts)
// Return:
//   minimum value in array

double** allocate_2D_array_double(const int N_x, const int N_y);
// Allocates 2D array as a continuous block in memory.
// Arguments:
//   N_x number of entries first index
//   N_y number of entries second index
// Return:
//   Allocated array

void free_2D_array_double(double **A);
// Frees a previously allocated 2D array
// Arguments:
//   A the array to be released

double*** allocate_3D_array_double(const int N_x, const int N_y, const int N_z);
// Allocates 3D array as a continuous block in memory.
// Arguments:
//   N_x number of entries first index
//   N_y number of entries second index
//   N_z number of entries third index
// Return:
//   Allocated array

void free_3D_array_double(double ***A);
// Frees a previously allocated 3D array
// Arguments:
//   A the array to be released

char*** allocate_3D_array_char(const int N_x, const int N_y, const int N_z);
// Allocates 3D array as a continuous block in memory.
// Arguments:
//   N_x number of entries first index
//   N_y number of entries second index
//   N_z number of entries third index
// Return:
//   Allocated array

void free_3D_array_char(char ***A);
// Frees a previously allocated 3D array
// Arguments:
//   A the array to be released

double**** allocate_4D_array_double(const int N_w, const int N_x, const int N_y, const int N_z);
// Allocates 4D array as a continuous block in memory.
// Arguments:
//   N_w number of entries in the first index
//   N_x number of entries second index
//   N_y number of entries third index
//   N_z number of entries fourth index
// Return:
//   Allocated array

void free_4D_array_double(double ****A);
// Frees a previously allocated 4D array
// Arguments:
//   A the array to be released

double***** allocate_5D_array_double(const int N_v, const int N_w, const int N_x,
  const int N_y, const int N_z);
// Allocates 5D array as a continuous block in memory.
// Arguments:
//   N_v number of entries in the first index
//   N_w number of entries in the second index
//   N_x number of entries third index
//   N_y number of entries fourth index
//   N_z number of entries fifth index
// Return:
//   Allocated array

void free_5D_array_double(double *****A);
// Frees a previously allocated 5D array
// Arguments:
//   A the array to be released

#endif
