// Boundary.c

#include "boundary.h"

void set_ghosts(grid *g, int layer)
{
  int i_g, i_y, i_z;
  
  #if DIMS_ > 1 // otherwise only ghosts at x-axis and x not looped over
  int i_x;
  #endif
  
  // ghosts along x_min. We will always limit parallelization to the outermost
  // layer. Also, lower dimensions will not have ghost cells along all edges.
  #if DIMS_ == 3
  #if parallelization_method_ == 1
  #pragma omp parallel for private (i_y, i_g)
  #elif parallelization_method_ == 2
  #pragma acc parallel loop private (i_y, i_g) present (g) collapse(3)
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
    #pragma omp parallel for private (i_g)
    #elif parallelization_method_ == 2
    #pragma acc parallel loop private (i_g) present (g) collapse(2)
    #endif
    #endif
    #if DIMS_ > 1
    for (i_y = ghosts_; i_y < GRES_y_ - ghosts_; i_y++)
    #else
    i_y = 0;
    #endif
    {
      #if parallelization_method_ == 2 && DIMS_ == 1
      #pragma acc parallel loop present (g)
      #endif
      for (i_g = 0; i_g < ghosts_; i_g++)
      {
        #if x_min_boundary_ == 0 // outflow boundary 

          g->q_flat[id5(layer, i_rho_, i_z, i_y, i_g)] =
            g->q_flat[id5(layer, i_rho_, i_z, i_y, ghosts_)];
          g->q_flat[id5(layer, i_S_x_, i_z, i_y, i_g)] = 
            +g->q_flat[id5(layer, i_S_x_, i_z, i_y, ghosts_)];
          #if DIMS_ > 1
          g->q_flat[id5(layer, i_S_y_, i_z, i_y, i_g)] = 
            g->q_flat[id5(layer, i_S_y_, i_z, i_y, ghosts_)];
          #endif
          #if DIMS_ > 2
          g->q_flat[id5(layer, i_S_z_, i_z, i_y, i_g)] = 
            g->q_flat[id5(layer, i_S_z_, i_z, i_y, ghosts_)];
          #endif
          g->q_flat[id5(layer, i_E_, i_z, i_y, i_g)] =
            g->q_flat[id5(layer, i_E_, i_z, i_y, ghosts_)];
          g->q_flat[id5(layer, i_tr1_, i_z, i_y, i_g)] =
            g->q_flat[id5(layer, i_tr1_, i_z, i_y, ghosts_)];

        #elif x_min_boundary_ == 1 // reflecting
          g->q_flat[id5(layer, i_rho_, i_z, i_y, i_g)] =
            g->q_flat[id5(layer, i_rho_, i_z, i_y, ghosts_)];
          g->q_flat[id5(layer, i_S_x_, i_z, i_y, i_g)] = 
            -g->q_flat[id5(layer, i_S_x_, i_z, i_y, ghosts_)];
          #if DIMS_ > 1
          g->q_flat[id5(layer, i_S_y_, i_z, i_y, i_g)] = 
            g->q_flat[id5(layer, i_S_y_, i_z, i_y, ghosts_)];
          #endif
          #if DIMS_ > 2
          g->q_flat[id5(layer, i_S_z_, i_z, i_y, i_g)] = 
            g->q_flat[id5(layer, i_S_z_, i_z, i_y, ghosts_)];
          #endif
          g->q_flat[id5(layer, i_E_, i_z, i_y, i_g)] =
            g->q_flat[id5(layer, i_E_, i_z, i_y, ghosts_)];
          g->q_flat[id5(layer, i_tr1_, i_z, i_y, i_g)] =
            g->q_flat[id5(layer, i_tr1_, i_z, i_y, ghosts_)];

        #elif x_min_boundary_ == 2 // periodic
          g->q_flat[id5(layer, i_rho_, i_z, i_y, i_g)] =
            g->q_flat[id5(layer, i_rho_, i_z, i_y, RES_x_ + i_g)];
          g->q_flat[id5(layer, i_S_x_, i_z, i_y, i_g)] = 
            +g->q_flat[id5(layer, i_S_x_, i_z, i_y, RES_x_ + i_g)];
          #if DIMS_ > 1
          g->q_flat[id5(layer, i_S_y_, i_z, i_y, i_g)] = 
            g->q_flat[id5(layer, i_S_y_, i_z, i_y, RES_x_ + i_g)];
          #endif
          #if DIMS_ > 2
          g->q_flat[id5(layer, i_S_z_, i_z, i_y, i_g)] = 
            g->q_flat[id5(layer, i_S_z_, i_z, i_y, RES_x_ + i_g)];
          #endif
          g->q_flat[id5(layer, i_E_, i_z, i_y, i_g)] =
            g->q_flat[id5(layer, i_E_, i_z, i_y, RES_x_ + i_g)];
          g->q_flat[id5(layer, i_tr1_, i_z, i_y, i_g)] =
            g->q_flat[id5(layer, i_tr1_, i_z, i_y, RES_x_ + i_g)];
        #endif
      }
    }
  }

  //----------------------------------------------------------------------------

  // ghosts along x_max. We will always limit parallelization to the outermost
  // layer. Also, lower dimensions will not have ghost cells along all edges.
  #if DIMS_ == 3
  #if parallelization_method_ == 1
  #pragma omp parallel for private (i_y, i_g)
  #elif parallelization_method_ == 2
  #pragma acc parallel loop private (i_y, i_g) present (g) collapse(3)
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
    #pragma omp parallel for private (i_g)
    #elif parallelization_method_ == 2
    #pragma acc parallel loop private (i_g) present (g) collapse(2)
    #endif
    #endif
    #if DIMS_ > 1
    for (i_y = ghosts_; i_y < GRES_y_ - ghosts_; i_y++)
    #else
    i_y = 0;
    #endif
    {
      #if parallelization_method_ == 2 && DIMS_ == 1
      #pragma acc parallel loop private (i_g) present (g)
      #endif
      for (i_g = 0; i_g < ghosts_; i_g++)
      {
        #if x_max_boundary_ == 0 // outflow boundary
          g->q_flat[id5(layer, i_rho_, i_z, i_y, GRES_x_ - i_g - 1)] = 
            g->q_flat[id5(layer, i_rho_, i_z, i_y, GRES_x_ - ghosts_ - 1)];
          g->q_flat[id5(layer, i_S_x_, i_z, i_y, GRES_x_ - i_g - 1)] = 
            +g->q_flat[id5(layer, i_S_x_, i_z, i_y, GRES_x_ - ghosts_ - 1)];
          #if DIMS_ > 1
          g->q_flat[id5(layer, i_S_y_, i_z, i_y, GRES_x_ - i_g - 1)] = 
            g->q_flat[id5(layer, i_S_y_, i_z, i_y, GRES_x_ - ghosts_ - 1)];
          #endif
          #if DIMS_ > 2
          g->q_flat[id5(layer, i_S_z_, i_z, i_y, GRES_x_ - i_g - 1)] = 
            g->q_flat[id5(layer, i_S_z_, i_z, i_y, GRES_x_ - ghosts_ - 1)];
          #endif
          g->q_flat[id5(layer, i_E_, i_z, i_y, GRES_x_ - i_g - 1)] = 
            g->q_flat[id5(layer, i_E_, i_z, i_y, GRES_x_ - ghosts_ - 1)];
          g->q_flat[id5(layer, i_tr1_, i_z, i_y, GRES_x_ - i_g - 1)] = 
            g->q_flat[id5(layer, i_tr1_, i_z, i_y, GRES_x_ - ghosts_ - 1)];
        #elif x_max_boundary_ == 1 // reflecting boundary
          g->q_flat[id5(layer, i_rho_, i_z, i_y, GRES_x_ - i_g - 1)] = 
            g->q_flat[id5(layer, i_rho_, i_z, i_y, GRES_x_ - ghosts_ - 1)];
          g->q_flat[id5(layer, i_S_x_, i_z, i_y, GRES_x_ - i_g - 1)] = 
            -g->q_flat[id5(layer, i_S_x_, i_z, i_y, GRES_x_ - ghosts_ - 1)];
          #if DIMS_ > 1
          g->q_flat[id5(layer, i_S_y_, i_z, i_y, GRES_x_ - i_g - 1)] = 
            g->q_flat[id5(layer, i_S_y_, i_z, i_y, GRES_x_ - ghosts_ - 1)];
          #endif
          #if DIMS_ > 2
          g->q_flat[id5(layer, i_S_z_, i_z, i_y, GRES_x_ - i_g - 1)] = 
            g->q_flat[id5(layer, i_S_z_, i_z, i_y, GRES_x_ - ghosts_ - 1)];
          #endif
          g->q_flat[id5(layer, i_E_, i_z, i_y, GRES_x_ - i_g - 1)] = 
            g->q_flat[id5(layer, i_E_, i_z, i_y, GRES_x_ - ghosts_ - 1)];
          g->q_flat[id5(layer, i_tr1_, i_z, i_y, GRES_x_ - i_g - 1)] = 
            g->q_flat[id5(layer, i_tr1_, i_z, i_y, GRES_x_ - ghosts_ - 1)];
          //printf("set x_maxx boundary zone: [%i][%i][%i] from [%i][%i][%i]\n", i_z, i_y, GRES_x_ - i_g - 1, i_z, i_y,GRES_x_ - ghosts_ - 1); fflush(stdout);
        #elif x_max_boundary_ == 2 // periodic
          g->q_flat[id5(layer, i_rho_, i_z, i_y, GRES_x_ - i_g - 1)] = 
            g->q_flat[id5(layer, i_rho_, i_z, i_y, 2 * ghosts_ - 1 - i_g)];
          g->q_flat[id5(layer, i_S_x_, i_z, i_y, GRES_x_ - i_g - 1)] = 
            g->q_flat[id5(layer, i_S_x_, i_z, i_y, 2 * ghosts_ - 1 - i_g)];
          #if DIMS_ > 1
          g->q_flat[id5(layer, i_S_y_, i_z, i_y, GRES_x_ - i_g - 1)] = 
            g->q_flat[id5(layer, i_S_y_, i_z, i_y, 2 * ghosts_ - 1 - i_g)];
          #endif
          #if DIMS_ > 2
          g->q_flat[id5(layer, i_S_z_, i_z, i_y, GRES_x_ - i_g - 1)] = 
            g->q_flat[id5(layer, i_S_z_, i_z, i_y, 2 * ghosts_ - 1 - i_g)];
          #endif
          g->q_flat[id5(layer, i_E_, i_z, i_y, GRES_x_ - i_g - 1)] = 
            g->q_flat[id5(layer, i_E_, i_z, i_y, 2 * ghosts_ - 1 - i_g)];
          g->q_flat[id5(layer, i_tr1_, i_z, i_y, GRES_x_ - i_g - 1)] = 
            g->q_flat[id5(layer, i_tr1_, i_z, i_y, 2 * ghosts_ - 1 - i_g)];
        #endif
      }
    }
  }

  //----------------------------------------------------------------------------

  // ghosts along y_min. We'll loop over the full x entries, including ghost
  // zones. This also set the corner ghosts that will not impact any actual
  // grid cells, but avoids compiler warnings about array values remaining
  // undetermined. The computational cost is negligible
  #if DIMS_ > 1
  
  #if DIMS_ == 3
  #if parallelization_method_ == 1
  #pragma omp parallel for private (i_g, i_x)
  #elif parallelization_method_ == 2
  #pragma acc parallel loop private (i_g, i_x) present(g) collapse(3)
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
    #pragma acc parallel loop private (i_x) present(g) collapse(2)
    #endif
    #endif
    for (i_g = 0; i_g < ghosts_; i_g++)
    {
      for (i_x = 0; i_x < GRES_x_; i_x++)
      {
        #if y_min_boundary_ == 0 // outflow boundary 
          g->q_flat[id5(layer, i_rho_, i_z, i_g, i_x)] =
            g->q_flat[id5(layer, i_rho_, i_z, ghosts_, i_x)];
          g->q_flat[id5(layer, i_S_x_, i_z, i_g, i_x)] = 
            g->q_flat[id5(layer, i_S_x_, i_z, ghosts_, i_x)];
          g->q_flat[id5(layer, i_S_y_, i_z, i_g, i_x)] = 
            +g->q_flat[id5(layer, i_S_y_, i_z, ghosts_, i_x)];
          #if DIMS_ > 2
          g->q_flat[id5(layer, i_S_z_, i_z, i_g, i_x)] = 
            g->q_flat[id5(layer, i_S_z_, i_z, ghosts_, i_x)];
          #endif
          g->q_flat[id5(layer, i_E_, i_z, i_g, i_x)] =
            g->q_flat[id5(layer, i_E_, i_z, ghosts_, i_x)];
          g->q_flat[id5(layer, i_tr1_, i_z, i_g, i_x)] =
            g->q_flat[id5(layer, i_tr1_, i_z, ghosts_, i_x)];
        #elif y_min_boundary_ == 1 // reflecting
          g->q_flat[id5(layer, i_rho_, i_z, i_g, i_x)] =
            g->q_flat[id5(layer, i_rho_, i_z, ghosts_, i_x)];
          g->q_flat[id5(layer, i_S_x_, i_z, i_g, i_x)] = 
            g->q_flat[id5(layer, i_S_x_, i_z, ghosts_, i_x)];
          g->q_flat[id5(layer, i_S_y_, i_z, i_g, i_x)] = 
            -g->q_flat[id5(layer, i_S_y_, i_z, ghosts_, i_x)];
          #if DIMS_ > 2
          g->q_flat[id5(layer, i_S_z_, i_z, i_g, i_x)] = 
            g->q_flat[id5(layer, i_S_z_, i_z, ghosts_, i_x)];
          #endif
          g->q_flat[id5(layer, i_E_, i_z, i_g, i_x)] =
            g->q_flat[id5(layer, i_E_, i_z, ghosts_, i_x)];
          g->q_flat[id5(layer, i_tr1_, i_z, i_g, i_x)] =
            g->q_flat[id5(layer, i_tr1_, i_z, ghosts_, i_x)];
          //printf("set y boundary zone: [%i][%i][%i] from [%i][%i][%i]\n", i_z, i_g, i_x, i_z, ghosts_, i_x); fflush(stdout);
        #elif y_min_boundary_ == 2 // periodic
          g->q_flat[id5(layer, i_rho_, i_z, i_g, i_x)] =
            g->q_flat[id5(layer, i_rho_, i_z, RES_y_ + i_g, i_x)];
          g->q_flat[id5(layer, i_S_x_, i_z, i_g, i_x)] = 
            +g->q_flat[id5(layer, i_S_x_, i_z, RES_y_ + i_g, i_x)];
          g->q_flat[id5(layer, i_S_y_, i_z, i_g, i_x)] = 
            g->q_flat[id5(layer, i_S_y_, i_z, RES_y_ + i_g, i_x)];
          #if DIMS_ > 2
          g->q_flat[id5(layer, i_S_z_, i_z, i_g, i_x)] = 
            g->q_flat[id5(layer, i_S_z_, i_z, RES_y_ + i_g, i_x)];
          #endif
          g->q_flat[id5(layer, i_E_, i_z, i_g, i_x)] =
            g->q_flat[id5(layer, i_E_, i_z, RES_y_ + i_g, i_x)];
          g->q_flat[id5(layer, i_tr1_, i_z, i_g, i_x)] =
            g->q_flat[id5(layer, i_tr1_, i_z, RES_y_ + i_g, i_x)];
        #endif
      }
    }
  }

  //----------------------------------------------------------------------------

  // ghosts along y_max. We'll loop over the full x entries, including ghost
  // zones. This also set the corner ghosts that will not impact any actual
  // grid cells, but avoids compiler warnings about array values remaining
  // undetermined. The computational cost is negligible

  #if DIMS_ == 3
  #if parallelization_method_ == 1
  #pragma omp parallel for private (i_g, i_x)
  #elif parallelization_method_ == 2
  #pragma acc parallel loop private (i_g, i_x) present (g) collapse(3)
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
    for (i_g = 0; i_g < ghosts_; i_g++)
    {
      for (i_x = 0; i_x < GRES_x_; i_x++)
      {
        #if y_max_boundary_ == 0 // outflow boundary
          g->q_flat[id5(layer, i_rho_, i_z, GRES_y_ - i_g - 1, i_x)] = 
            g->q_flat[id5(layer, i_rho_, i_z, GRES_y_ - ghosts_ - 1, i_x)];
          g->q_flat[id5(layer, i_S_x_, i_z, GRES_y_ - i_g - 1, i_x)] = 
            g->q_flat[id5(layer, i_S_x_, i_z, GRES_y_ - ghosts_ - 1, i_x)];
          g->q_flat[id5(layer, i_S_y_, i_z, GRES_y_ - i_g - 1, i_x)] = 
            +g->q_flat[id5(layer, i_S_y_, i_z, GRES_y_ - ghosts_ - 1, i_x)];
          #if DIMS_ > 2
          g->q_flat[id5(layer, i_S_z_, i_z, GRES_y_ - i_g - 1, i_x)] = 
            g->q_flat[id5(layer, i_S_z_, i_z, GRES_y_ - ghosts_ - 1, i_x)];
          #endif
          g->q_flat[id5(layer, i_E_, i_z, GRES_y_ - i_g - 1, i_x)] = 
            g->q_flat[id5(layer, i_E_, i_z, GRES_y_ - ghosts_ - 1, i_x)];
          g->q_flat[id5(layer, i_tr1_, i_z, GRES_y_ - i_g - 1, i_x)] = 
            g->q_flat[id5(layer, i_tr1_, i_z, GRES_y_ - ghosts_ - 1, i_x)];
        #elif y_max_boundary_ == 1 // reflecting boundary
          g->q_flat[id5(layer, i_rho_, i_z, GRES_y_ - i_g - 1, i_x)] = 
            g->q_flat[id5(layer, i_rho_, i_z, GRES_y_ - ghosts_ - 1, i_x)];
          g->q_flat[id5(layer, i_S_x_, i_z, GRES_y_ - i_g - 1, i_x)] = 
            g->q_flat[id5(layer, i_S_x_, i_z, GRES_y_ - ghosts_ - 1, i_x)];
          g->q_flat[id5(layer, i_S_y_, i_z, GRES_y_ - i_g - 1, i_x)] = 
            -g->q_flat[id5(layer, i_S_y_, i_z, GRES_y_ - ghosts_ - 1, i_x)];
          #if DIMS_ > 2
          g->q_flat[id5(layer, i_S_z_, i_z, GRES_y_ - i_g - 1, i_x)] = 
            g->q_flat[id5(layer, i_S_z_, i_z, GRES_y_ - ghosts_ - 1, i_x)];
          #endif
          g->q_flat[id5(layer, i_E_, i_z, GRES_y_ - i_g - 1, i_x)] = 
            g->q_flat[id5(layer, i_E_, i_z, GRES_y_ - ghosts_ - 1, i_x)];
          g->q_flat[id5(layer, i_tr1_, i_z, GRES_y_ - i_g - 1, i_x)] = 
            g->q_flat[id5(layer, i_tr1_, i_z, GRES_y_ - ghosts_ - 1, i_x)];
          //printf("set y_max boundary zone: [%i, %i, %i] from [%i, %i, %i]\n", i_z, GRES_x_ - i_g - 1, i_x, i_z,GRES_y_ - ghosts_ - 1, i_x); fflush(stdout);
        #elif y_max_boundary_ == 2 // periodic
          g->q_flat[id5(layer, i_rho_, i_z, GRES_y_ - i_g - 1, i_x)] = 
            g->q_flat[id5(layer, i_rho_, i_z, 2 * ghosts_ - 1 - i_g, i_x)];
          g->q_flat[id5(layer, i_S_x_, i_z, GRES_y_ - i_g - 1, i_x)] = 
            g->q_flat[id5(layer, i_S_x_, i_z, 2 * ghosts_ - 1 - i_g, i_x)];
          g->q_flat[id5(layer, i_S_y_, i_z, GRES_y_ - i_g - 1, i_x)] = 
            g->q_flat[id5(layer, i_S_y_, i_z, 2 * ghosts_ - 1 - i_g, i_x)];
          #if DIMS_ > 2
          g->q_flat[id5(layer, i_S_z_, i_z, GRES_y_ - i_g - 1, i_x)] = 
            g->q_flat[id5(layer, i_S_z_, i_z, 2 * ghosts_ - 1 - i_g, i_x)];
          #endif
          g->q_flat[id5(layer, i_E_, i_z, GRES_y_ - i_g - 1, i_x)] = 
            g->q_flat[id5(layer, i_E_, i_z, 2 * ghosts_ - 1 - i_g, i_x)];
          g->q_flat[id5(layer, i_tr1_, i_z, GRES_y_ - i_g - 1, i_x)] = 
            g->q_flat[id5(layer, i_tr1_, i_z, 2 * ghosts_ - 1 - i_g, i_x)];
        #endif
      }
    }
  }

  #endif // DIMS_ > 1

  //----------------------------------------------------------------------------

  #if DIMS_ > 2
  
  // ghosts along z_min. We'll loop over the full x,y entries, including ghost
  // zones. This also set the corner ghosts that will not impact any actual
  // grid cells, but avoids compiler warnings about array values remaining
  // undetermined. The computational cost is negligible. We'll parallelize the
  // loop over y, since there are likely fewer ghosts than cores
  
  for (i_g = 0; i_g < ghosts_; i_g++)
  {
    #if parallelization_method_ == 1
    #pragma omp parallel for private (i_x)
    #elif parallelization_method_ == 2
    #pragma acc parallel loop private (i_x) present (g) collapse(2)
    #endif
    for (i_y = 0; i_y < GRES_y_; i_y++)
    {
      for (i_x = 0; i_x < GRES_x_; i_x++)
      {
        #if z_min_boundary_ == 0 // outflow boundary 
          g->q_flat[id5(layer, i_rho, i_g, i_y, i_x)] =
            g->q_flat[id5(layer, i_rho_, ghosts_, i_y, i_x)];
          g->q_flat[id5(layer, i_S_x_, i_g, i_y, i_x)] = 
            g->q_flat[id5(layer, i_S_x, ghosts_, i_y, i_x)];
          g->q_flat[id5(layer, i_S_y_, i_g, i_y, i_x)] = 
            g->q_flat[id5(layer, i_S_y_, ghosts_, i_y, i_x)];
          g->q_flat[id5(layer, i_S_z_, i_g, i_y, i_x)] = 
            +g->q_flat[id5(layer, i_S_z_, ghosts_, i_y, i_x)];
          g->q_flat[id5(layer, i_E_, i_g, i_y, i_x)] =
            g->q_flat[id5(layer, i_E_, ghosts_, i_y, i_x)];
          g->q_flat[id5(layer, i_tr1_, i_g, i_y, i_x)] =
            g->q_flat[id5(layer, i_tr1_, ghosts_, i_y, i_x)];
        #elif z_min_boundary_ == 1 // reflecting
          g->q_flat[id5(layer, i_rho_, i_g, i_y, i_x)] =
            g->q_flat[id5(layer, i_rho_, ghosts_, i_y, i_x)];
          g->q_flat[id5(layer, i_S_x_, i_g, i_y, i_x)] = 
            g->q_flat[id5(layer, i_S_x_, ghosts_, i_y, i_x)];
          g->q_flat[id5(layer, i_S_y_, i_g, i_y, i_x)] = 
            g->q_flat[id5(layer, i_S_y_, ghosts_, i_y, i_x)];
          g->q_flat[id5(layer, i_S_z_, i_g, i_y, i_x)] = 
            -g->q_flat[id5(layer, i_S_z_, ghosts_, i_y, i_x)];
          g->q_flat[id5(layer, i_E_, i_g, i_y, i_x)] =
            g->q_flat[id5(layer, i_E_, ghosts_, i_y, i_x)];
          g->q_flat[id5(layer, i_tr1_, i_g, i_y, i_x)] =
            g->q_flat[id5(layer, i_tr1_, ghosts_, i_y, i_x)];
        #elif z_min_boundary_ == 2 // periodic
          g->q_flat[id5(layer, i_rho_, i_g, i_y, i_x)] =
            g->q_flat[id5(layer, i_rho_, RES_z_ + i_g, i_y, i_x)];
          g->q_flat[id5(layer, i_S_x_, i_g, i_y, i_x)] = 
            +g->q_flat[id5(layer, i_S_x_, RES_z_ + i_g, i_y, i_x)];
          g->q_flat[id5(layer, i_S_y_, i_g, i_y, i_x)] = 
            g->q_flat[id5(layer, i_S_y_, RES_z_ + i_g, i_y, i_x)];
          g->q_flat[id5(layer, i_S_z_, i_g, i_y, i_x)] = 
            g->q_flat[id5(layer, i_S_z_, RES_z_ + i_g, i_y, i_x)];
          g->q_flat[id5(layer, i_E_, i_g, i_y, i_x)] =
            g->q_flat[id5(layer, i_E_, RES_z_ + i_g, i_y, i_x)];
          g->q_flat[id5(layer, i_tr1_, i_g, i_y, i_x)] =
            g->q_flat[id5(layer, i_tr1_, RES_z_ + i_g, i_y, i_x)];
        #endif
      }
    }
  }

  //----------------------------------------------------------------------------

  // ghosts along z_max. We'll loop over the full x entries, including ghost
  // zones. This also set the corner ghosts that will not impact any actual
  // grid cells, but avoids compiler warnings about array values remaining
  // undetermined. The computational cost is negligible

  for (i_g = 0; i_g < ghosts_; i_g++)
  {
    #if parallelization_method_ == 1
    #pragma omp parallel for private (i_x)
    #elif parallelization_method_ == 2
    #pragma acc parallel loop private (i_x) present (g) collapse(2)
    #endif
    for (i_y = 0; i_y < GRES_y_; i_y++)
    {
      for (i_x = 0; i_x < GRES_x_; i_x++)
      {
        #if z_max_boundary_ == 0 // outflow boundary
          g->q_flat[id5(layer, i_rho_, GRES_z_ - i_g - 1, i_y, i_x)] = 
            g->q_flat[id5(layer, i_rho_, GRES_z_ - ghosts_ - 1, i_y, i_x)];
          g->q_flat[id5(layer, i_S_x_, GRES_z_ - i_g - 1, i_y, i_x)] = 
            g->q_flat[id5(layer, i_S_x_, GRES_z_ - ghosts_ - 1, i_y, i_x)];
          g->q_flat[id5(layer, i_S_y_, GRES_z_ - i_g - 1, i_y, i_x)] = 
            g->q_flat[id5(layer, i_S_y_, GRES_z_ - ghosts_ - 1, i_y, i_x)];
          g->q_flat[id5(layer, i_S_z_, GRES_z_ - i_g - 1, i_y, i_x)] = 
            +g->q_flat[id5(layer, i_S_z_, GRES_z_ - ghosts_ - 1, i_y, i_x)];
          g->q_flat[id5(layer, i_E_, GRES_z_ - i_g - 1, i_y, i_x)] = 
            g->q_flat[id5(layer, i_E_, GRES_z_ - ghosts_ - 1, i_y, i_x)];
          g->q_flat[id5(layer, i_tr1_, GRES_z_ - i_g - 1, i_y, i_x)] = 
            g->q_flat[id5(layer, i_tr1_, GRES_z_ - ghosts_ - 1, i_y, i_x)];
        #elif z_max_boundary_ == 1 // reflecting boundary
          g->q_flat[id5(layer, i_rho_, GRES_z_ - i_g - 1, i_y, i_x)] = 
            g->q_flat[id5(layer, i_rho_, GRES_z_ - ghosts_ - 1, i_y, i_x)];
          g->q_flat[id5(layer, i_S_x_, GRES_z_ - i_g - 1, i_y, i_x)] = 
            g->q_flat[id5(layer, i_S_x_, GRES_z_ - ghosts_ - 1, i_y, i_x)];
          g->q_flat[id5(layer, i_S_y_, GRES_z_ - i_g - 1, i_y, i_x)] = 
            g->q_flat[id5(layer, i_S_y_, GRES_z_ - ghosts_ - 1, i_y, i_x)];
          g->q_flat[id5(layer, i_S_z_, GRES_z_ - i_g - 1, i_y, i_x)] = 
            -g->q_flat[id5(layer, i_S_z_, GRES_z_ - ghosts_ - 1, i_y, i_x)];
          g->q_flat[id5(layer, i_E_, GRES_z_ - i_g - 1, i_y, i_x)] = 
            g->q_flat[id5(layer, i_E_, GRES_z_ - ghosts_ - 1, i_y, i_x)];
          g->q_flat[id5(layer, i_tr1_, GRES_z_ - i_g - 1, i_y, i_x)] = 
            g->q_flat[id5(layer, i_tr1_, GRES_z_ - ghosts_ - 1, i_y, i_x)];
        #elif z_max_boundary_ == 2 // periodic
          g->q_flat[id5(layer, i_rho_, GRES_z_ - i_g - 1, i_y, i_x)] = 
            g->q_flat[id5(layer, i_rho_, 2 * ghosts_ - 1 - i_g, i_y, i_x)];
          g->q_flat[id5(layer, i_S_x_, GRES_z_ - i_g - 1, i_y, i_x)] = 
            g->q_flat[id5(layer, i_S_x_, 2 * ghosts_ - 1 - i_g, i_y, i_x)];
          g->q_flat[id5(layer, i_S_y_, GRES_z_ - i_g - 1, i_y, i_x)] = 
            g->q_flat[id5(layer, i_S_y_, 2 * ghosts_ - 1 - i_g, i_y, i_x)];
          g->q_flat[id5(layer, i_S_z_, GRES_z_ - i_g - 1, i_y, i_x)] = 
            g->q_flat[id5(layer, i_S_z_, 2 * ghosts_ - 1 - i_g, i_y, i_x)];
          g->q_flat[id5(layer, i_E_, GRES_z_ - i_g - 1, i_y, i_x)] = 
            g->q_flat[id5(layer, i_E_, 2 * ghosts_ - 1 - i_g, i_y, i_x)];
          g->q_flat[id5(layer, i_tr1_, GRES_z_ - i_g - 1, i_y, i_x)] = 
            g->q_flat[id5(layer, i_tr1_, 2 * ghosts_ - 1 - i_g, i_y, i_x)];
        #endif
      }
    }
  }
  #endif // DIMS_ > 2

}

////////////////////////////////////////////////////////////////////////////////
