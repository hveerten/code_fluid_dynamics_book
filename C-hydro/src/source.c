#include "source.h"

void set_source(grid *g, int from)
{
  
  double grav[DIMS_];
  
  grav[0] = 0.;
  #if DIMS_ > 1
  grav[1] = -0.1;
  #endif
  #if DIMS_ > 2
  grav[2] = 0.;
  #endif
  
  int i_x, i_y, i_z; // loop over grid cells
  
  //double dx = (g->x_max - g->x_min) / (RES_x_); // for position dependent 2D gravity
  //double dy = (g->y_max - g->y_min) / (RES_y_); // for position dependent 2D gravity

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

      //double y = g->y_min + (i_y - ghosts_ + 0.5) * dy; // for position dependent 2D gravity

      #if parallelization_method_ == 2 && DIMS_ == 1
      #pragma acc parallel loop present(g)
      #endif
      for (i_x = ghosts_; i_x < GRES_x_ - ghosts_; i_x++)
      {  
        
        //double x = g->x_min + (i_x - ghosts_ + 0.5) * dx; // for position dependent 2D gravity
  
        //double MG = 0.1 * 1; // gravitational constant times black hole mass
        //double r = sqrt(x*x + y * y);
        //double G_scale = MG / (r * r);
        //if (r > 0.1)
        //{
        //  grav[0] = -G_scale * (x / r);
        //  grav[1] = -G_scale * (y / r);
        //}
        //printf("grav[0] = %e, grav[1] = %e\n", grav[0], grav[1]);
        
        g->source_flat[id4(i_rho_, i_z, i_y, i_x)] = 0.;
        
        g->source_flat[id4(i_S_x_, i_z, i_y, i_x)] = grav[0] *
          g->q_flat[id5(from, i_rho_, i_z, i_y, i_x)];

        g->source_flat[id4(i_E_, i_z, i_y, i_x)] =
          grav[0] * g->q_flat[id5(from, i_S_x_, i_z, i_y, i_x)];

        #if DIMS_ > 1
        g->source_flat[id4(i_S_y_, i_z, i_y, i_x)] = grav[1] *
          g->q_flat[id5(from, i_rho_, i_z, i_y, i_x)];
        g->source_flat[id4(i_E_, i_z, i_y, i_x)] +=
          grav[1] * g->q_flat[id5(from, i_S_y_, i_z, i_y, i_x)];
        #endif

        #if DIMS_ > 2
        g->source_flat[id4(i_S_z_, i_z, i_y, i_x)] = grav[2] *
          g->q_flat[id5(from, i_rho_, i_z, i_y, i_x)];
        g->source_flat[id4(i_E_, i_z, i_y, i_x)] +=
          grav[2] * g->q_flat[id5(from, i_S_z_, i_z, i_y, i_x)];
        #endif

        g->source_flat[id4(i_tr1_, i_z, i_y, i_x)] = 0.;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

void apply_source(grid *g, int layer)
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
    #elif parallelization_method == 2
    #pragma acc parallel loop private (i_x) present(g) collapse(2)
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
          g->q_flat[id5(layer, i_var, i_z, i_y, i_x)] += 
            g->dt * g->source_flat[id4(i_var, i_z, i_y, i_x)];
        }
      }
    }
  }
}
