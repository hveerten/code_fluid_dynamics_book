// Boundary.h - routines to deal with the boundary conditions

#ifndef BOUNDARY_
#define BOUNDARY_

#include <stdio.h> // this include only used during debugging, in practice
#include "grid.h"
#include "settings.h"
#include "arraytools.h"

void set_ghosts(grid *g, int layer);
// Set all the ghost cell values
// Arguments:
//  g      pointer to grid to operate on
//  layer  which layer to set (multiple layers used for higher order time step)

#endif
