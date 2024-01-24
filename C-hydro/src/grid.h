// grid.h containing the grid struct used by most programs
// As well as all relevant switches and settings

#ifndef GRID_
#define GRID_

#include <stdbool.h>
#include <stdint.h>
#include "settings.h"

// array index labels, based in part on `DIMS_' set in settings.h
#define i_rho_   0
#define i_S_x_   1
#define i_S_y_   2
#define i_S_z_   3
#define i_E_     (1 + DIMS_)
#define i_v_x_   1
#define i_v_y_   2
#define i_v_z_   3
#define i_p_     (1 + DIMS_)
#define i_tr1_   (2 + DIMS_) // tracer variable
#define no_vars_ (3 + DIMS_)
#define x_dir_   0
#define y_dir_   1
#define z_dir_   2

typedef struct grid
{
  double *****q; // state vector grid (5 dimensional)
  double ****pr; // primitive values grid (4 dimensional)
  double ****F_x; // x direction fluxes (4 dimensional)
  #if DIMS_ > 1
  double ****F_y; // y direction fluxes (4 dimensional)
  #endif
  #if DIMS_ > 2
  double ****F_z; // z direction fluxes (4 dimensional)
  #endif
  double ****source; // source term (4 dimensional)
  double ***dt_grid; // used when determining timestep (3 dimensional)

  uint32_t i_t; // number of iterations
  double t; // current grid time
  double dt; // time step size
  double x_min; // grid min x coordinate
  double x_max; // grid max x coordinate
  double y_min; // grid min x coordinate
  double y_max; // grid max x coordinate
  double z_min; // grid min z coordinate
  double z_max; // grid max z coordinate
  double t_max;  // end time for the run

  double dt_snapshot; // linear time between snapshots. Negative if not used
  
  // for openacc, we will work with flattened versions of the arrays
  // on the device. Luckily, these are included within the pointer-arithmetic
  // multidimensional array allocation. We just need to define a set of labels
  // for them such that the compiler does not get confused by the syntax.
  // To avoid duplicating code, we'll include flat versions in all
  // parallelization_method options.
  // btw, in 2022 open_acc could handle up to 2D arrays as dynamically allocated
  // struct members, but failed at 3D and higher, at least in my testing
  
  double *q_flat;
  double *pr_flat;
  double *F_x_flat;
  #if DIMS_ > 1
  double *F_y_flat; // y direction fluxes (4 dimensional)
  #endif
  #if DIMS_ > 2
  double *F_z_flat; // z direction fluxes (4 dimensional)
  #endif
  double *source_flat; // source term (4 dimensional)
  double *dt_grid_flat; // used when determining timestep (3 dimensional)

  char *title; // a text string to name the initial conditions
} grid;

// further resolution macros implementing ghost cell settings from settings.h
#if disable_RK_
#define layers_ 1
#else
#define layers_ 3
#endif

#if DIMS_ == 1
#define GRES_x_ (RES_x_ + 2 * ghosts_)
#define GRES_y_ 1
#define GRES_z_ 1
#endif

#if DIMS_ == 2
#define GRES_x_ (RES_x_ + 2 * ghosts_)
#define GRES_y_ (RES_y_ + 2 * ghosts_)
#define GRES_z_ 1
#endif

#if DIMS_ == 3
#define GRES_x_ (RES_x_ + 2 * ghosts_)
#define GRES_y_ (RES_y_ + 2 * ghosts_)
#define GRES_z_ (RES_z_ + 2 * ghosts_)
#endif

// macros for indexing of flat arrays. This only works for arrays
// where the ordering of the entries is layers, var, z, y, x, with smaller-
// dimensional arrays missing entries counting from the left (ie y, x for 2D)
#define id2(iy, ix) ((iy) * GRES_x_ + ix)
#define id3(iz, iy, ix) ((iz) * GRES_y_ * GRES_x_ + (iy) * GRES_x_ + ix)
#define id4(iv, iz, iy, ix) ((iv) * GRES_z_ * GRES_y_ * GRES_x_ + \
  (iz) * GRES_y_ * GRES_x_ + (iy) * GRES_x_ + ix)
#define id5(il, iv, iz, iy, ix) ((il) * no_vars_ * GRES_z_ * GRES_y_ * GRES_x_ + \
  (iv) * GRES_z_ * GRES_y_ * GRES_x_ + \
  (iz) * GRES_y_ * GRES_x_ + (iy) * GRES_x_ + ix)

#endif
