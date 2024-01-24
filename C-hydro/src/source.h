// source.h

#include "settings.h"
#include "grid.h"

void set_source(grid *g, int from);
// Sets the source terms across the grid, ghost cells excluded
// Arguments:
//   g     pointer to the grid
//   from  The layer from which to take the source terms (higher order time
//         step)

void apply_source(grid *g, int layer);
// Applies the source terms across the grid, ghost cells excluded
// Arguments:
//   g       pointer to the grid
//   layer   the layer to which the source terms are to be applied

