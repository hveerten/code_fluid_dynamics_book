// toy-hydro. A simple fixed-mesh C-based hydrodynamics solver

#include <stdbool.h>
#include <stdio.h>

#include "arraytools.h"
#include "grid.h"
#include "initial_conditions.h"
#include "consprim.h"
#include "solver.h"
#include "IO.h"
#include "boundary.h"
#include "settings.h"
#include "source.h"

////////////////////////////////////////////////////////////////////////////////

void initialize(grid *g)
{
  g->q = allocate_5D_array_double(layers_, no_vars_, GRES_z_, GRES_y_, GRES_x_);
  g->pr = allocate_4D_array_double(no_vars_, GRES_z_, GRES_y_, GRES_x_);
  g->F_x = allocate_4D_array_double(no_vars_, GRES_z_, GRES_y_, GRES_x_);
  #if DIMS_ > 1
  g->F_y = allocate_4D_array_double(no_vars_, GRES_z_, GRES_y_, GRES_x_);
  #endif
  #if DIMS_ > 2
  g->F_z = allocate_4D_array_double(no_vars_, GRES_z_, GRES_y_, GRES_x_);
  #endif
  g->source = allocate_4D_array_double(no_vars_, GRES_z_, GRES_y_, GRES_x_);
  g->dt_grid = allocate_3D_array_double(GRES_z_, GRES_y_, GRES_x_);

  // print grid initialization settings  
  printf("# x resolution: = %i\n", RES_x_);
  #if DIMS_ > 1
  printf("# y resolution: = %i\n", RES_y_);
  #endif
  #if DIMS_ > 2
  printf("# z resolution: = %i\n", RES_z_);
  #endif
  
  // clean out state vectors. Maybe not necessary, but for RK stepping methods
  // our current implementation always updates a q with a weighted contribution
  // from all layers. Even if the weights are zero for entries, this might still
  // trigger a warning when zero x undetermined is added in the very first
  // iteration. So we turn that into zero x zero.
  int i_x, i_y, i_z, i_vars, i_layer;
  for (i_z = 0; i_z < GRES_z_; i_z++)
  {
    for (i_y = 0; i_y < GRES_y_; i_y++)
    {
      for (i_x = 0; i_x < GRES_x_; i_x++)
      {
        for (i_vars = 0; i_vars < no_vars_; i_vars++)
        {
          for (i_layer = 0; i_layer < layers_; i_layer++)
          {
            g->q[i_layer][i_vars][i_z][i_y][i_x] = 0.0;
          }
        }
      }
    }
  }

  // set up the flat version labels
  g->q_flat = g->q[0][0][0][0];
  g->pr_flat = g->pr[0][0][0];
  g->F_x_flat = g->F_x[0][0][0];
  #if DIMS_ > 1
  g->F_y_flat = g->F_y[0][0][0];
  #endif  
  #if DIMS_ > 2
  g->F_z_flat = g->F_z[0][0][0];
  #endif  
  g->source_flat = g->source[0][0][0];
  g->dt_grid_flat = g->dt_grid[0][0];
  
  #if parallelization_method_ == 2

  // get the data onto the device
  #pragma acc enter data copyin(g[0:1])
  #pragma acc enter data copyin(g->q_flat[0:layers_ * no_vars_ * GRES_z_ * GRES_y_ * GRES_x_])
  #pragma acc enter data create(g->pr_flat[0:no_vars_ * GRES_z_ * GRES_y_ * GRES_x_])
  #pragma acc enter data create(g->F_x_flat[0:no_vars_ * GRES_z_ * GRES_y_ * GRES_x_])
  #if DIMS_ > 1
  #pragma acc enter data create(g->F_y_flat[0:no_vars_ * GRES_z_ * GRES_y_ * GRES_x_])
  #endif
  #if DIMS_ > 2
  #pragma acc enter data create(g->F_z_flat[0:no_vars_ * GRES_z_ * GRES_y_ * GRES_x_])
  #endif
  #pragma acc enter data create(g->source_flat[0:no_vars_ * GRES_z_ * GRES_y_ * GRES_x_])
  #pragma acc enter data create(g->dt_grid_flat[0:GRES_z_ * GRES_y_ * GRES_x_])

  #endif
}

////////////////////////////////////////////////////////////////////////////////

void free_memory(grid *g)
{
  // wrap up the GPU memory versions
  #if parallelization_method_ == 2
  #pragma acc exit data delete(g->q_flat)
  #pragma acc exit data delete(g->pr_flat)
  #pragma acc exit data delete(g->F_x_flat)
  #if DIMS_ > 1
  #pragma acc exit data delete(g->F_y_flat)
  #endif
  #if DIMS_ > 2
  #pragma acc exit data delete(g->F_z_flat)
  #endif
  #pragma acc exit data delete(g->source_flat)
  #pragma acc exit data delete(g->dt_grid_flat)
  #pragma acc exit data delete(g)
  #endif

  free_5D_array_double(g->q);
  free_4D_array_double(g->pr);
  free_4D_array_double(g->F_x);
  #if DIMS_ > 1
  free_4D_array_double(g->F_y);
  #endif
  #if DIMS_ > 2
  free_4D_array_double(g->F_z);
  #endif
  free_4D_array_double(g->source);
  free_3D_array_double(g->dt_grid);

}

////////////////////////////////////////////////////////////////////////////////

void apply(grid *g, int RK)
{
  int i_x, i_y, i_z; // loop over grid cells

  double dx = (g->x_max - g->x_min) / (RES_x_);
  #if DIMS_ > 1
  double dy = (g->y_max - g->y_min) / (RES_y_);
  #endif
  #if DIMS_ > 2
  double dz = (g->z_max - g->z_min) / (RES_z_);
  #endif

  // declare variables for values differing per RK step
  int i_to;
  double Cq0, Cq1, Cq2;
  double Cfs;

  // set the weight values for the different RK steps of 3rd order RK method
  
  if (RK == -1) // revert to forward Euler
  {
    i_to = 0;
    Cq0 = 1.;
    Cq1 = 0.;
    Cq2 = 0.;
    Cfs = 1.;
  }
  
  if (RK == 0)
  {
    i_to = 1;
    Cq0 = 1.;
    Cq1 = 0.;
    Cq2 = 0.;
    Cfs = 1.;
  }
  
  if (RK == 1)
  {
    i_to = 2;
    Cq0 = 3./4.;
    Cq1 = 1./4.;
    Cq2 = 0.;
    Cfs = 1./4.;
  }
  
  if (RK == 2)
  {
    i_to = 0;
    Cq0 = 1./3.;
    Cq1 = 0.;
    Cq2 = 2./3.;
    Cfs = 2./3.;
  }
  
  //----------------------------------------------------------------------------

  // loop over the cells
  #if DIMS_ == 3
  #if parallelization_method_ == 1
  #pragma omp parallel for private (i_x, i_y)
  #elif parallelization_method_ == 2
  #pragma acc parallel loop private (i_x, i_y) present (g) collapse(3)
  #endif
  #endif
  #if DIMS_ > 2
  for (i_z = ghosts_; i_z < GRES_z_ - ghosts_; i_z++)
  #else
  i_z = 0;
  #endif
  {
    #if DIMS_ == 2
    #if parallelization_method_ == 1
    #pragma omp parallel for private (i_x)
    #elif parallelization_method_ == 2
    #pragma acc parallel loop private (i_x) present (g) collapse(2)
    #endif
    #endif
    #if DIMS_ > 1
    for (i_y = ghosts_; i_y < GRES_y_ - ghosts_; i_y++)
    #else
    i_y = 0;
    #endif
    {
      #if parallelization_method_ == 2 && DIMS_ == 1
      #pragma acc parallel loop present(g)
      #endif
      for (i_x = ghosts_; i_x < GRES_x_ - ghosts_; i_x++)
      {  
        int i_var;
        for (i_var = 0; i_var < no_vars_; i_var++)
        {
          int i_out = id5(i_to, i_var, i_z, i_y, i_x);
          g->q_flat[i_out] = Cq0 * g->q_flat[id5(0, i_var, i_z, i_y, i_x)];

          g->q_flat[i_out] += Cq1 * g->q_flat[id5(1, i_var, i_z, i_y, i_x)];

          g->q_flat[i_out] += Cq2 * g->q_flat[id5(2, i_var, i_z, i_y, i_x)];

          g->q_flat[i_out] += Cfs * g->dt / dx *
            (g->F_x_flat[id4(i_var, i_z, i_y, i_x)] - g->F_x_flat[id4(i_var, i_z, i_y, i_x+1)]);
          
          #if DIMS_ > 1
          g->q_flat[i_out] += Cfs * g->dt / dy *
            (g->F_y_flat[id4(i_var, i_z, i_y, i_x)] - g->F_y_flat[id4(i_var, i_z, i_y+1, i_x)]);
          #endif
          
          #if DIMS_ > 2
          g->q_flat[i_out] += Cfs * g->dt / dz *
            (g->F_z_flat[id4(i_var, i_z, i_y, i_x)] - g->F_z_flat[id4(i_var, i_z+1, i_y, i_x)]);
          #endif
          
          if (use_source_) g->q_flat[i_out] += Cfs * g->dt * 
              g->source_flat[id4(i_var, i_z, i_y, i_x)];
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

void apply_floor(grid *g)
{
  int i_x, i_y, i_z; // loop over grid cells
  
  #if DIMS_ == 3
  #if parallelization_method_ == 1
  #pragma omp parallel for private (i_x, i_y)
  #elif parallelization_method_ == 2
  #pragma acc parallel loop private (i_x, i_y) present (g) collapse(3)
  #endif
  #endif
  #if DIMS_ > 2
  for (i_z = ghosts_; i_z < GRES_z_ - ghosts_; i_z++)
  #else
  i_z = 0;
  #endif
  {
    #if DIMS_ == 2
    #if parallelization_method_ == 1
    #pragma omp parallel for private (i_x)
    #elif parallelization_method_ == 2
    #pragma acc parallel loop private (i_x) present (g) collapse(2)
    #endif
    #endif
    #if DIMS_ > 1
    for (i_y = ghosts_; i_y < GRES_y_ - ghosts_; i_y++)
    #else
    i_y = 0;
    #endif
    {
      #if parallelization_method_ == 2 && DIMS_ == 1
      #pragma acc parallel loop present(g)
      #endif
      for (i_x = ghosts_; i_x < GRES_x_ - ghosts_; i_x++)
      { 
        int i_to = id5(0, i_rho_, i_z, i_y, i_x); 
        g->q_flat[i_to] = fmax(rho_floor_, g->q_flat[i_to]);
        i_to = id5(0, i_E_, i_z, i_y, i_x); 
        g->q_flat[i_to] =
          fmax(E_floor_, g->q_flat[i_to]);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

void print_header_message(grid *p_g, FILE* p_out)
{
  // print various opening messages, title of program and of simulation run

  fprintf(p_out, "# toy-hydro\n");
  fprintf(p_out, "# running set-up labeled: %s\n", p_g->title);
  fprintf(p_out, "# Adiabatic exponent set to: %f\n", GAMMA_AD_);
  
  // print messages about the system (settings) running the simulation and
  fprintf(p_out, "#--------------------------------------------------------\n");
  #if parallelization_method_ == 1
  int max_threads = omp_get_max_threads(); // get maximum possible number of
  fprintf(p_out, "# This run has %i CPU threads available to work with.\n",
    max_threads);
  #endif
  fprintf(p_out, "# Parallelization set to : %i\n",
    parallelization_method_);

  long mem_use;
  mem_use = layers_ * no_vars_ * GRES_x_ * GRES_y_ * GRES_z_; // q
  mem_use += no_vars_ * GRES_x_ * GRES_y_ * GRES_z_; // pr
  mem_use += no_vars_ * GRES_x_ * GRES_y_ * GRES_z_; // F_x
  #if DIMS_ > 1
  mem_use += no_vars_ * GRES_x_ * GRES_y_ * GRES_z_; // F_y
  #endif
  #if DIMS_ > 2
  mem_use += no_vars_ * GRES_x_ * GRES_y_ * GRES_z_; // F_z
  #endif
  mem_use += no_vars_ * GRES_x_ * GRES_y_ * GRES_z_; // source
  mem_use += no_vars_ * GRES_x_ * GRES_y_ * GRES_z_; // dt_grid

  mem_use = mem_use * sizeof(double);
  
  fprintf(p_out, "# Total memory usage of the grid arrays: %ld bytes, %ld kB,"
    " %ld MB, %ld GB\n", mem_use, mem_use / 1024, mem_use / 1024 / 1024, 
    mem_use / 1024 / 1024 / 1024);

  fprintf(p_out, "#--------------------------------------------------------\n");

}

////////////////////////////////////////////////////////////////////////////////

int main()
{
  bool stoprun;
  grid g;
  int i_snapshot = 0; // current snapshot data dump
  char filename[200];
  double t_snapshot; // upcoming snapshot dump time
  
  initialize(&g); // allocate memory
  
  set_grid_initial_conditions(&g);
  
  // prep for running
  g.i_t = 0; // set current number of iterations to zero
  stoprun = false;
  t_snapshot = 0. + g.dt_snapshot;
  
  print_header_message(&g, stdout);
  
  //stoprun = true;
  
  if (!use_source_) printf("# Source term disabled.\n");
  
  // save an initial state data dump
  sprintf(filename, "out%04d.txt", i_snapshot);
  fprint_state(filename, &g);
  #if verbose_ != 0
    printf("# Saving snapshot %s at 0 iterations. Gridt time = %e.\n", filename, 
      g.t);
  #endif
  i_snapshot++;

  // launch the run
  while(!stoprun)
  {
    if (disable_RK_)
    {
      // first order forward Euler
      set_ghosts(&g, 0);
      cons2prim(&g, 0);

      set_dt(&g, CFL_factor_);
      g.dt = fmax(dt_floor, g.dt);
      // cap the timestep if needed
      if (g.t + g.dt > g.t_max)
      {
        g.dt = (g.t_max - g.t) * 1.000000001;
        stoprun = true;
      }

      if (g.t + g.dt > t_snapshot)
      {
        // avoid overshooting a snapshot dump time by too much
        g.dt = (t_snapshot - g.t) * 1.000000001;
      }

      #if parallelization_method_ == 2
      #pragma acc update device(g.dt)
      #endif

      set_F_x(&g);
      #if DIMS > 1
      set_F_y(&g);
      #endif
      #if DIMS_ > 2
      set_F_z(&g);
      #endif
      if (use_source_) set_source(&g, 0);

      apply(&g, -1); // forward Euler step, y flux
      
    }
    else
    {
      // 3rd order Runge-Kutta
    
      // first step
      set_ghosts(&g, 0);
      cons2prim(&g, 0);

      // first RK timestep, also obtain dt value
      
      set_dt(&g, 0.3);
      g.dt = fmax(dt_floor, g.dt);
      
      // cap the timestep if needed
      if (g.t + g.dt > g.t_max)
      {
        g.dt = (g.t_max - g.t) * 1.0000000000001;
        stoprun = true;
      }
      else if (g.t + g.dt > t_snapshot)
      {
        // avoid overshooting a snapshot dump time by too much
        g.dt = (t_snapshot - g.t) * 1.0000000000001;
      }
      
      #if parallelization_method_ == 2
      #pragma acc update device(g.dt)
      #endif

      set_F_x(&g);
      #if DIMS_ > 1
      set_F_y(&g);
      #endif
      #if DIMS_ > 2
      set_F_z(&g);
      #endif
      if (use_source_) set_source(&g, 0);
      
      apply(&g, 0); // first RK step
    
      // second step
      set_ghosts(&g, 1);
      cons2prim(&g, 1);
      set_F_x(&g);
      #if DIMS_ > 1
      set_F_y(&g);
      #endif
      #if DIMS_ > 2
      set_F_z(&g);
      #endif
      if (use_source_) set_source(&g, 1);
      apply(&g, 1); // second RK step
    
      // third step
      set_ghosts(&g, 2);
      cons2prim(&g, 2);
      set_F_x(&g);
      #if DIMS_ > 1
      set_F_y(&g);
      #endif
      #if DIMS_ > 2
      set_F_z(&g);
      #endif      
      if (use_source_) set_source(&g, 2);
      apply(&g, 2); // third RK step
      
      // apply floor
      //apply_floor(&g);   
    }
    
    // increment the time
    g.t += g.dt;
    #if parallelization_method_ == 2
    #pragma acc update device(g.t)
    #endif
    if (g.t >= t_snapshot)
    {
      cons2prim(&g, 0);
      sprintf(filename, "out%04d.txt", i_snapshot);
      fprint_state(filename, &g);

      #if verbose_ != 0
        printf("# Saving snapshot %s at %u iterations. Gridt time = %e, "
          "grid dt = %e.\n", filename, g.i_t, g.t, g.dt);
      #endif
      t_snapshot += g.dt_snapshot;
      i_snapshot++;
    }
   
    g.i_t++; // complete another iteration;
    if (g.i_t >= it_max_) stoprun = true;
    #if parallelization_method_ == 2
    #pragma acc update device(g.i_t)
    #endif

    if (g.i_t % 100 == 0)
      printf("%u, %e, %e\n", g.i_t, g.t, g.dt);
  } 

  cons2prim(&g, 0);

  free_memory(&g);
  
  return 0;
}
