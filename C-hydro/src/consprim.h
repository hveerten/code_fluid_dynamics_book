// consprim routines to go from conserved to primitive variables and back

#ifndef CONSPRIM_
#define CONSPRIM_

#include <stdio.h>

#include "grid.h"
#include "settings.h"

void cons2prim(grid *g, int layer);
// computes all the primitive variables on the grid based on conservatives.
// This function does include the ghost cells
// arguments:
//   g     pointer to the grid
//   layer which layer of the state vector to compute from. The layers
//         store different intermediate stages of a higher-order timestep

void prim2cons(grid *g);
// computes all the conservative variables on the grid based on primitives.
// This function does not compute ghost cells values. It always stores the
// outcome in layer 0
// arguments:
//   g pointer to the grid

#if parallelization_method_ == 2
#pragma acc routine
#endif
void cons2prim_local(double *q, double *pr);
// Computes the local primitive variables based on a local state vector
// Arguments:
//   q   state vector
// Returns:
//   pr  primitive values vector

#if parallelization_method_ == 2
#pragma acc routine
#endif
void prim2cons_local(double *pr, double *q);
// Computes the local conserved variables based on a primitive state vector
// Arguments:
//   pr  primitive vector
// Returns:
//   q   state vector

#endif
