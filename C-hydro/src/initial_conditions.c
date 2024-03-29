// initial_conditions.h
#include <math.h>
#include <stdio.h>

#include "initial_conditions.h"

void set_grid_initial_conditions(grid *g)
{
  int i_x, i_y, i_z;
  
  g->t = 0;
  
  #if fluid_problem_ == 0 // A shock-tube problem
  
    g->t_max = 0.2;
    g->title = "basic shock tube";

    g->x_min = 0.;
    g->x_max = 1.;
    #if DIMS_ > 1
      g->y_min = 0.;
      g->y_max = 1.;
    #endif
    #if DIMS_ > 2
      g->z_min = 0.;
      g->z_max = 1.;
    #endif

  #endif 
  
  #if fluid_problem_ == 1 // A Kelvin-Helmholtz problem

    g->t_max = 0.8;
    g->title = "Kelvin-Helmholtz instability";

    #if DIMS_ < 2
    #error "not enough dimensions for a KH problem"
    #endif
    
    g->x_min = 0.;
    g->x_max = 1.;
    g->y_min = 0.;
    g->y_max = 1.;
    
    #if DIMS_ > 2
      g->z_min = 0.;
      g->z_max = 1.;
    #endif
    
  #endif
  
  #if fluid_problem_ == 2 // A Rayleigh-Taylor problem

    g->t_max = 12.75;
    g->title = "Rayleigh-Taylor instability";

    #if DIMS_ < 2
    #error "not enough dimensions for a RT problem"
    #endif
    
    g->x_min = -0.75;
    g->x_max = 0.75;
    g->y_min = -0.75;
    g->y_max = 0.75;
    
    #if DIMS_ > 2
      g->z_min = -0.25;
      g->z_max = 0.25;  
    #endif
  
  #endif
  
  #if fluid_problem_ == 3 // a localized gravity source problem

    g->t_max = 3.;
    g->title = "gravity ball";
  
    g->x_min = -1.;
    g->x_max = 1.;

    #if DIMS_ > 1
      g->y_min = -1.;
      g->y_max = 1.;
    #endif
    
    #if DIMS_ > 2
      g->z_min = -1.;
      g->z_max = 1.;
    #endif
  
  #endif 
    
  g->dt_snapshot = g->t_max / (no_snapshots_ - 1);
  
  #if parallelization_method_ == 2
  // get the updated grid info into the GPU memory
  #pragma acc update device(g->x_min, g->x_max, g->y_min, g->y_max, g->z_min)
  #pragma acc update device(g->z_max, g->t_max, g->dt_snapshot, g->t)
  #endif
  
  double dx = (g->x_max - g->x_min) / (RES_x_);
  #if DIMS_ > 1
    double dy = (g->y_max - g->y_min) / (RES_y_);
  #endif
  #if DIMS_ > 2
    double dz = (g->z_max - g->z_min) / (RES_z_);
  #endif

  // the loop over the grid during initialization is NOT done in parallel. This
  // makes it easier to implement initial conditions using an algorithm
  // where cell values depend on each other
  
  #if DIMS_ > 2
  for (i_z = ghosts_; i_z < GRES_z_ - ghosts_; i_z++)
  #else
  i_z = 0;
  #endif
  {
    #if DIMS_ > 2
    double z = g->z_min + (i_z - ghosts_ + 0.5) * dz;
    #endif

    #if DIMS_ > 1
    for (i_y = ghosts_; i_y < GRES_y_ - ghosts_; i_y++)
    #else
    i_y = 0;
    #endif
    {
      #if DIMS_ > 1
      double y = g->y_min + (i_y - ghosts_ + 0.5) * dy;
      #endif

      for (i_x = ghosts_; i_x < GRES_x_ - ghosts_; i_x++)
      {  
        double x; // cell-centered coordinates
        x = g->x_min + (i_x - ghosts_ + 0.5) * dx;

        #if fluid_problem_ == 0         // set up some shock tube

          #if DIMS_ == 1
          if (x < 0.3)
          #endif
          #if DIMS_ == 2
          if (x < 0.3 && y < 0.3)
          #endif
          #if DIMS_ == 3
          if (x < 0.3 && y < 0.3 && z < 0.3)
          #endif
          {
            g->pr[i_rho_][i_z][i_y][i_x] = 1.;
            g->pr[i_v_x_][i_z][i_y][i_x] = 0.75;
            #if DIMS_ > 1
              g->pr[i_v_y_][i_z][i_y][i_x] = 0.;
            #endif
            #if DIMS_ > 2
              g->pr[i_v_z_][i_z][i_y][i_x] = 0.;
            #endif
            g->pr[i_p_][i_z][i_y][i_x] = 1.;
            g->pr[i_tr1_][i_z][i_y][i_x] = 1.;
          }
          else
          {
            g->pr[i_rho_][i_z][i_y][i_x] = 0.125;
            g->pr[i_v_x_][i_z][i_y][i_x] = 0.;
            #if DIMS_ > 1
              g->pr[i_v_y_][i_z][i_y][i_x] = 0.;
            #endif
            #if DIMS_ > 2
              g->pr[i_v_z_][i_z][i_y][i_x] = 0.;
            #endif
            g->pr[i_p_][i_z][i_y][i_x] = 0.1;
            g->pr[i_tr1_][i_z][i_y][i_x] = 0.;
          }

        #endif
                      
        #if fluid_problem_ == 1
        
          // Kelvin-Helmholtz
          // further parameters: periodic boundaries, box 0..1 in x and y dirs
          //    adiabatic index 7/5

          double w_0 = 0.1; double sigma = 0.05 / sqrt(2.0);
          if (y >= 0.25 && y < 0.75)
          {
            g->pr[i_rho_][i_z][i_y][i_x] = 2;
            g->pr[i_v_x_][i_z][i_y][i_x] = 0.5;
            g->pr[i_v_y_][i_z][i_y][i_x] = w_0 * sin(4. * M_PI * x) * 
              (exp(-(y-0.25)*(y-0.25)/(2.*sigma*sigma)) + 
              exp(-(y-0.75)*(y-0.75)/(2.*sigma*sigma)));
            #if DIMS_ > 2
              g->pr[i_v_z_][i_z][i_y][i_x] = 0.;
            #endif
            g->pr[i_p_][i_z][i_y][i_x] = 2.5;
            g->pr[i_tr1_][i_z][i_y][i_x] = 1.;         
          }
          else
          {
            g->pr[i_rho_][i_z][i_y][i_x] = 1;
            g->pr[i_v_x_][i_z][i_y][i_x] = -0.5;
            g->pr[i_v_y_][i_z][i_y][i_x] = w_0 * sin(4. * M_PI * x) * 
              (exp(-(y-0.25)*(y-0.25)/(2.*sigma*sigma)) + 
              exp(-(y-0.75)*(y-0.75)/(2.*sigma*sigma)));
            #if DIMS_ > 2
              g->pr[i_v_z_][i_z][i_y][i_x] = 0.;
            #endif
            g->pr[i_p_][i_z][i_y][i_x] = 2.5;
            g->pr[i_tr1_][i_z][i_y][i_x] = 0.;         
          }
        
        #endif

        #if fluid_problem_ == 2 // Rayleigh-Taylor
        
          if (y < 0.)
          {
            g->pr[i_rho_][i_z][i_y][i_x] = 1.;
            g->pr[i_v_x_][i_z][i_y][i_x] = +0.25;
            #if DIMS_ == 2
              g->pr[i_v_y_][i_z][i_y][i_x] = +0.1 * (1. + cos(4. * M_PI * x)) * 
                (1. + cos (3. * M_PI * y)) / 4.;
            #endif
            #if DIMS_ > 2
              g->pr[i_v_y_][i_z][i_y][i_x] =
                0.1 * (1. + cos(4. * M_PI * x)) * 
                (1. + cos(4. * M_PI * z)) * (1. + cos (3. * M_PI * y)) / 4.;
              g->pr[i_v_z_][i_z][i_y][i_x] = 0.;
            #endif
            g->pr[i_p_][i_z][i_y][i_x] = 
              2.5 - 0.1 * g->pr[i_rho_][i_z][i_y][i_x] * y;
            g->pr[i_tr1_][i_z][i_y][i_x] = 1.;
          }
          else
          {
            g->pr[i_rho_][i_z][i_y][i_x] = 2.;
            g->pr[i_v_x_][i_z][i_y][i_x] = -0.25;
            #if DIMS_ == 2
              g->pr[i_v_y_][i_z][i_y][i_x] = +0.1 * (1. + cos(4. * M_PI * x)) * 
               (1. + cos (3. * M_PI * y)) / 4.;
            #endif
            #if DIMS_ > 2
              g->pr[i_v_y_][i_z][i_y][i_x] = 0.1 * (1. + cos(4. * M_PI * x)) * 
                (1. + cos(4. * M_PI * z)) * (1. + cos (3. * M_PI * y)) / 4.;
              g->pr[i_v_z_][i_z][i_y][i_x] = 0;
            #endif
            g->pr[i_p_][i_z][i_y][i_x] =
              2.5 - 0.1 * g->pr[i_rho_][i_z][i_y][i_x] * y;
            g->pr[i_tr1_][i_z][i_y][i_x] = 0.;
          }
        
        #endif
        
        #if fluid_problem_ == 3 // gravity ball
        
          double y0 = 0.5;
          double x0 = -0.5;
          double r0 = 0.1;
      
          if ((y - y0) * (y-y0) + (x - x0) * (x-x0) < r0*r0)
          {
            g->pr[i_rho_][i_z][i_y][i_x] = 10.;
            g->pr[i_v_x_][i_z][i_y][i_x] = 0.1;
            #if DIMS_ == 2
              g->pr[i_v_y_][i_z][i_y][i_x] = 0.;
            #endif
            #if DIMS_ > 2
              g->pr[i_v_y_][i_z][i_y][i_x] = 0.;
              g->pr[i_v_z_][i_z][i_y][i_x] = 0.;
            #endif
            g->pr[i_p_][i_z][i_y][i_x] = 0.1; // or whatever
            g->pr[i_tr1_][i_z][i_y][i_x] = 1.;
          }
          else
          {
            g->pr[i_rho_][i_z][i_y][i_x] = 1.;
            g->pr[i_v_x_][i_z][i_y][i_x] = 0;
            #if DIMS_ == 2
              g->pr[i_v_y_][i_z][i_y][i_x] = 0;
            #endif
            #if DIMS_ > 2
              g->pr[i_v_y_][i_z][i_y][i_x] = 0;
              g->pr[i_v_z_][i_z][i_y][i_x] = 0;
            #endif
            g->pr[i_p_][i_z][i_y][i_x] = 0.1; // or whatever
            g->pr[i_tr1_][i_z][i_y][i_x] = 0.;
          }
        
        #endif
      }
    }
  }

  #if parallelization_method_ == 2
  // get the updated grid info into the GPU (device) memory
  #pragma acc update device(g->pr_flat[0:no_vars_ * GRES_z_ * GRES_y_ * GRES_x_])
  #endif
    
  // In this case, we've set initial conditions from the primitive variables,
  // so we need to compute the conserved ones.

  prim2cons(g);
}
