// solver.h

// HLLC solver (Toro page 331)

#ifndef SOLVER_H_
#define SOLVER_H_

#include <math.h>
#include <stdbool.h>

#include "arraytools.h"
#include "grid.h"
#include "settings.h"
#include "reconstruct.h"

void set_dt(grid *g, double scale);
// sets the timestep using the CFL condition. It is part of solver to allow
// for tweaks using wavespeeds (rather than just sound speed), that can be
// computed in a solver-algorithm dependent manner
// Arguments:
//   g      struct containing grid info
//   scale  safety factor to rescale dt (<0.3 typically)

void set_F_x(grid *g);
#if DIMS_ > 1
void set_F_y(grid *g);
#endif
#if DIMS_ > 2
void set_F_z(grid *g);
#endif

#endif // SOLVER_H_
