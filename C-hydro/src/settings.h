// settings.h
//
// change this file for problem-specific settings

#ifndef SETTINGS_H_
#define SETTINGS_H_

// Different options for code parallelization. 0 is none, 1 is openMP,
// 2 is openACC
#define parallelization_method_ 0

#define DIMS_   1 // number of dimensions
#define RES_x_  200 // x-direction resolution, not counting ghost zones
#define RES_y_  50 // y-direction resolution, not counting ghost zones
#define RES_z_  50 // z-direction resolution, not counting ghost zones
#define ghosts_ 2  // number of ghost cells on a side
#define it_max_ 1000000 // maximum number of iterations

#define fluid_problem_ 0 // a label to specify which set of initial conditions
  // is used, in case the user wishes to group more initial conditions together
  // in a single initial_conditions.c file. Can also be used te select
  // user-specified source terms and boundary conditions, if needed. 
  // The following fluid problems are provided out-of-the box:
  // 00  shock tube
  // 01  a Kelvin-Helmholtz problem (requires at least 2 dimensions)
  // 02  a Rayleigh-Taylor problem (requires at least 2 dimensions)
  // 03  a mock central gravity problem

#define no_snapshots_ 2 // how many data dumps to write to disc
#define verbose_ 1 // set to zero (FALSE) for less output to terminal

// boundary conditions
// 0 is outflow, 1 is reflecting, 2 is periodic, 3 is user (3 not implemented)
#define x_min_boundary_ 0
#define x_max_boundary_ 0
#define y_min_boundary_ 1
#define y_max_boundary_ 1
#define z_min_boundary_ 2
#define z_max_boundary_ 2

#define dt_floor   1e-9
#define p_floor_   1e-5
#define rho_floor_ 1e-5
#define E_floor_   1e-10

#define CFL_factor_ 0.3 // scale factor when setting CFL timestep 

// Start with fixed index polytropic gas
#define GAMMA_AD_ 1.4

// flags to revert to lowest order methods
#define disable_RK_ true // drop from 3rd order Runge-Kutta to first order 
 // forward Euler timestep if true
#define disable_PLM_ true // drop to piecewise constant if true
#define use_source_ 0 // 1 // non-zero number to use source

//------------------------------------------------------------------------------
// some quick checks on the settings above

#if DIMS_ < 1 || DIMS_ > 3
#error "settings.h ERROR: wrong number of dimensions"
#endif

#if parallelization_method_ < 0 || parallelization_method_ > 2
#error "settings.h ERROR: wrong setting parallelization_method"
#endif

#if RES_x_ < 1
#error "settings.h ERROR: not enough zones RES_x_ in x-dimension"
#endif

#if RES_y_ < 1 && DIMS_ > 1
#error "settings.h ERROR: not enough zones RES_y_ in y-dimension"
#endif 

#if RES_z_ < 1 && DIMS_ > 2
#error "settings.h ERROR: not enough zones RES_z_ in z-dimension"
#endif 

#if ghosts_ < 1
#error "settings.h ERROR: not enough ghost zones in ghosts_"
#endif

#if it_max_ < 1
#warning "settings.h WARNING: it_max set to very low number"
#endif

#if x_min_boundary_ < 0 || x_min_boundary_ > 3
#error "settings.h ERROR: x_min_boundary_ set to illegal value"
#endif

#if x_max_boundary_ < 0 || x_max_boundary_ > 3
#error "settings.h ERROR: x_max_boundary_ set to illegal value"
#endif

#if DIMS_ > 1 && (y_min_boundary_ < 0 || y_min_boundary_ > 3)
#error "settings.h ERROR: y_min_boundary_ set to illegal value"
#endif

#if DIMS_ > 1 && (y_max_boundary_ < 0 || y_max_boundary_ > 3)
#error "settings.h ERROR: y_max_boundary_ set to illegal value"
#endif

#if DIMS_ > 2 && (z_min_boundary_ < 0 || z_min_boundary_ > 3)
#error "settings.h ERROR: z_min_boundary_ set to illegal value"
#endif

#if DIMS_ > 2 && (z_max_boundary_ < 0 || z_max_boundary_ > 3)
#error "settings.h ERROR: z_max_boundary_ set to illegal value"
#endif

#if no_snapshots_ < 2
#error "settings.h ERROR: illegal value for no_snapshots_"
#endif

#endif // SETTINGS_H_
