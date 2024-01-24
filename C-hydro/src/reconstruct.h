// reconstruct.h
// This version implements the PLM, with a basic minmod slope limiter

#ifndef RECONSTRUCT_H_
#define RECONSTRUCT_H_

#include <math.h>
#include <stdio.h> // for debugging purposes

#include "grid.h"
#include "settings.h"
#include "consprim.h"

#ifndef sign_function_
#define sign_function_
#define sign(a) ( ( (a) < 0 )  ?  -1   : ( (a) > 0 ) ) 
#endif // probably move this to
  // some 'extramath' library or something

#if parallelization_method_ == 2
#pragma acc routine
#endif
void set_L_state(grid *g, int i_x, int i_y, int i_z, char dir, double *q_L, 
  double *pr_L);
// sets the state to the left of a cell boundary
// Arguments:
//   g     pointer to struct containing the grid data
//   i_x   cell boundary x entry number (cell boundary i is to left of cell i)
//   i_y   cell boundary y entry number (cell boundary i is to left of cell i)
//   i_z   cell boundary y entry number (cell boundary i is to left of cell i)
//   dir   0 for x-direction left state, 1 for y direction, 2 for z
// Return:
//   q_L   conserved values state vector
//   pr_L  primitive values state vector

#if parallelization_method_ == 2
#pragma acc routine
#endif
void set_R_state(grid *g, int i_x, int i_y, int i_z, char dir, 
  double *q_R, double *pr_R);
// sets the state to the right of a cell boundary
// Arguments:
//   g     pointer to struct containing the grid data
//   i_x   cell boundary x entry number (cell boundary i is to left of cell i)
//   i_y   cell boundary y entry number (cell boundary i is to left of cell i)
//   i_z   cell boundary z entry number (cell boundary i is to left of cell i)
//   dir   0 for x-direction left state, 1 for y, 2 for z
// Return:
//   q_R   conserved values state vector
//   pr_R  primitive values state vector

#endif // RECONSTRUCT_H_
