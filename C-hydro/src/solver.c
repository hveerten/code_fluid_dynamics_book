// solver.c

#include "solver.h"

void set_dt(grid *g, double scale)
{
  int i_x, i_y, i_z;

  // for the GPU version we will sort out all the reduction stuff within this
  // routine directly. struct members cannot be used in a reduction clause,
  // so we have a workaround using a temporary variable dt
  #if parallelization_method_ == 2
  double dt = 1e300;
  #endif

  double dx = (g->x_max - g->x_min) / (RES_x_);
  #if DIMS_ > 1
  double dy = (g->y_max - g->y_min) / (RES_y_);
  #endif
  #if DIMS_ > 2
  double dz = (g->z_max - g->z_min) / (RES_z_);
  #endif

  #if DIMS_ == 3
  #if parallelization_method_ == 1
  #pragma omp parallel for private (i_x, i_y)
  #elif parallelization_method_ == 2
  #pragma acc parallel loop private (i_x, i_y) present (g) reduction(min:dt) collapse(3)
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
    #pragma acc parallel loop private (i_x) present (g) reduction(min:dt) collapse(2)
    #endif
    #endif
    #if DIMS_ > 1
    for (i_y = ghosts_; i_y < GRES_y_ - ghosts_; i_y++)
    #else
    i_y = 0;
    #endif
    {
      #if parallelization_method_ == 2 && DIMS_ == 1
      #pragma acc parallel loop present (g) reduction(min:dt)
      #endif
      for (i_x = ghosts_; i_x < GRES_x_ - ghosts_; i_x++)
      {
        double c_s;
        int i_cor = id3(i_z, i_y, i_x);
        
        c_s  = sqrt(GAMMA_AD_ * g->pr_flat[id4(i_p_, i_z, i_y, i_x)] / 
          g->pr_flat[id4(i_rho_, i_z, i_y, i_x)]);
        
        g->dt_grid_flat[i_cor] = dx / 
          (c_s + fabs(g->pr_flat[id4(i_v_x_, i_z, i_y, i_x)]));
        
        #if DIMS_ > 1
        g->dt_grid_flat[i_cor] = fmin(g->dt_grid_flat[i_cor], 
          dy / (c_s + fabs(g->pr_flat[id4(i_v_y_, i_z, i_y, i_x)])));
        #endif
        
        #if DIMS_ > 2
        g->dt_grid_flat[i_cor] = fmin(g->dt_grid_flat[i_cor], 
          dz / (c_s + fabs(g->pr_flat[id4(i_v_z_, i_z, i_y, i_x)])));
        #endif
        
        #if parallelization_method_ == 2
        dt = fmin(g->dt_grid_flat[i_cor], dt);
        #endif
      }
    }
  }

  #if parallelization_method_ == 2
  
  g->dt = scale * dt;
  #pragma acc update device(g->dt)
  
  #else
   
  #if DIMS_ == 1
  g->dt = scale * find_minimum_3D(g->dt_grid, 1, 1, RES_x_, 0,
    0, ghosts_);
  #endif
  
  #if DIMS_ == 2
  g->dt = scale * find_minimum_3D(g->dt_grid, 1, RES_y_, RES_x_, 0,
    ghosts_, ghosts_);
  #endif
  
  #if DIMS_ == 3
  g->dt = scale * find_minimum_3D(g->dt_grid, RES_z_, RES_y_, RES_x_, ghosts_,
    ghosts_, ghosts_);
  #endif
  
  #endif
}

////////////////////////////////////////////////////////////////////////////////

void set_F_x(grid *g)
{
  // loop over all cell walls
  
  int i_x, i_y, i_z; // loop over grid cells
  
  #if DIMS_ == 3
  #if parallelization_method_ == 1
  #pragma omp parallel for private (i_x, i_y)
  #elif parallelization_method_ == 2
  #pragma acc parallel loop private (i_x, i_y) present(g) collapse(3)
  #endif
  #endif
  #if DIMS_ > 2
  for (i_z = ghosts_; i_z < GRES_z_ - ghosts_ + 1; i_z++)
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
    #if DIMS_ > 1
    for (i_y = ghosts_; i_y < GRES_y_ - ghosts_ + 1; i_y++)
    #else
    i_y = 0;
    #endif
    {
      #if parallelization_method_ == 2 && DIMS_ == 1
      #pragma acc parallel loop present (g)
      #endif
      for (i_x = ghosts_; i_x < GRES_x_ - ghosts_ + 1; i_x++)
      {
        int i_var; // loop over state vector entries

        double q_L[no_vars_];
        double pr_L[no_vars_];
        double F_L[no_vars_];
        double q_star_L[no_vars_];
    
        double q_R[no_vars_];
        double pr_R[no_vars_];
        double F_R[no_vars_];
        double q_star_R[no_vars_];
    
        double c_s_L, c_s_R; // sound speeds left and right
        double rho_bar, c_s_bar; // average density and sound speed across boundary
        double p_star; // inner region
      
        double S_L, S_R, S_star; // wave speeds (not momenta)

        // set the left state and flux
        
        set_L_state(g, i_x, i_y, i_z, x_dir_, q_L, pr_L);

        F_L[i_rho_] = q_L[i_rho_] * pr_L[i_v_x_];
        F_L[i_S_x_] = q_L[i_rho_] * pr_L[i_v_x_] * pr_L[i_v_x_] + pr_L[i_p_];
        #if DIMS_ > 1
        F_L[i_S_y_] = q_L[i_rho_] * pr_L[i_v_y_] * pr_L[i_v_x_];
        #endif
        #if DIMS_ > 2
        F_L[i_S_z_] = q_L[i_rho_] * pr_L[i_v_z_] * pr_L[i_v_x_];
        #endif
        F_L[i_E_] = (q_L[i_E_] + pr_L[i_p_]) * pr_L[i_v_x_];
        F_L[i_tr1_] = q_L[i_tr1_] * pr_L[i_v_x_];

        c_s_L = sqrt(GAMMA_AD_ * pr_L[i_p_] / pr_L[i_rho_]);

        // set the right state and flux
        set_R_state(g, i_x, i_y, i_z, x_dir_, q_R, pr_R);

        F_R[i_rho_] = q_R[i_rho_] * pr_R[i_v_x_];
        F_R[i_S_x_] = q_R[i_rho_] * pr_R[i_v_x_] * pr_R[i_v_x_] + pr_R[i_p_];
        #if DIMS_ > 1
        F_R[i_S_y_] = q_R[i_rho_] * pr_R[i_v_y_] * pr_R[i_v_x_];
        #endif
        #if DIMS_ > 2
        F_R[i_S_z_] = q_R[i_rho_] * pr_R[i_v_z_] * pr_R[i_v_x_];
        #endif
        F_R[i_E_] = (q_R[i_E_] + pr_R[i_p_]) * pr_R[i_v_x_];
        F_R[i_tr1_] = q_R[i_tr1_] * pr_R[i_v_x_];

        c_s_R = sqrt(GAMMA_AD_ * pr_R[i_p_] / pr_R[i_rho_]);

        // Apply the HLLC solver
        rho_bar = 0.5 * (pr_R[i_rho_] + pr_L[i_rho_]);
        c_s_bar = 0.5 * (c_s_L + c_s_R);
        p_star = 0.5  * (pr_L[i_p_] + pr_R[i_p_]) - 
          0.5 * (pr_R[i_v_x_] - pr_L[i_v_x_]) * rho_bar * c_s_bar;
        p_star = fmax(p_star, 0.);

        // HLL left wave speed estimate. Don't confuse wave speed 'S' with
        // momentum 'S'. We're just using the notation from Toro, 3rd ed. page 331
        if (p_star <= pr_L[i_p_])
        {
          S_L = pr_L[i_v_x_] - c_s_L;
        }
        else
        {
          S_L = pr_L[i_v_x_] - c_s_L * 
            sqrt(1. + (GAMMA_AD_ + 1.) / (2. * GAMMA_AD_) * 
              (p_star / pr_L[i_p_] - 1.));
        }
      
        // HLL right wave speed estimate
        if (p_star <= pr_R[i_p_])
        {
          S_R = pr_R[i_v_x_] + c_s_R;
        }
        else
        {
          S_R = pr_R[i_v_x_] + c_s_R *
            sqrt(1. + (GAMMA_AD_ + 1.) / (2. * GAMMA_AD_) * 
              (p_star / pr_R[i_p_] - 1.));
        }
      
        // Begin HLLC-specific estimates (up until here was just HLL)
        S_star = (pr_R[i_p_] - pr_L[i_p_] + 
          pr_L[i_rho_] * pr_L[i_v_x_] * (S_L - pr_L[i_v_x_]) -
          pr_R[i_rho_] * pr_R[i_v_x_] * (S_R - pr_R[i_v_x_])) /
          (pr_L[i_rho_] * (S_L - pr_L[i_v_x_]) - 
            pr_R[i_rho_] * (S_R - pr_R[i_v_x_]));

        // starred region state vector estimates, starting with left
        q_star_L[i_rho_] = pr_L[i_rho_] * (S_L - pr_L[i_v_x_]) / (S_L - S_star);

        q_star_L[i_S_x_] = pr_L[i_rho_] * (S_L - pr_L[i_v_x_]) / (S_L - S_star) *
          S_star;

        #if DIMS_ > 1
        q_star_L[i_S_y_] = pr_L[i_rho_] * (S_L - pr_L[i_v_x_]) / (S_L - S_star) *
          pr_L[i_v_y_]; // perpendicular momentum just advects, like rho
        #endif

        #if DIMS_ > 2
        q_star_L[i_S_z_] = pr_L[i_rho_] * (S_L - pr_L[i_v_x_]) / (S_L - S_star) *
          pr_L[i_v_z_]; // perpendicular momentum just advects, like rho
        #endif

        q_star_L[i_E_] = pr_L[i_rho_] * (S_L - pr_L[i_v_x_]) / (S_L - S_star) *
          (q_L[i_E_] / pr_L[i_rho_] + (S_star - pr_L[i_v_x_]) * 
            (S_star + pr_L[i_p_] / (pr_L[i_rho_] * (S_L - pr_L[i_v_x_]))));

        q_star_L[i_tr1_] = pr_L[i_tr1_] * (S_L - pr_L[i_v_x_]) / (S_L - S_star);

        // starred region state vector estimates, now the right
        q_star_R[i_rho_] = pr_R[i_rho_] * (S_R - pr_R[i_v_x_]) / (S_R - S_star);

        q_star_R[i_S_x_] = pr_R[i_rho_] * (S_R - pr_R[i_v_x_]) / (S_R - S_star) *
          S_star;

        #if DIMS_ > 1
        q_star_R[i_S_y_] = pr_R[i_rho_] * (S_R - pr_R[i_v_x_]) / (S_R - S_star) *
          pr_R[i_v_y_]; // perpendicular momentum just advects, like rho
        #endif

        #if DIMS_ > 2
        q_star_R[i_S_z_] = pr_R[i_rho_] * (S_R - pr_R[i_v_x_]) / (S_R - S_star) *
          pr_R[i_v_z_]; // perpendicular momentum just advects, like rho
        #endif

        q_star_R[i_E_] = pr_R[i_rho_] * (S_R - pr_R[i_v_x_]) / (S_R - S_star) *
          (q_R[i_E_] / pr_R[i_rho_] + (S_star - pr_R[i_v_x_]) * 
            (S_star + pr_R[i_p_] / (pr_R[i_rho_] * (S_R - pr_R[i_v_x_]))));  

        q_star_R[i_tr1_] = pr_R[i_tr1_] * (S_R - pr_R[i_v_x_]) / (S_R - S_star);
  
        // set the flux through the cell boundaries, four possibilities for the
        // for possible regions where the cell boundary can be found
        if (0 <= S_L) // whole wave pattern moved to right of cell boundary
        {
          for (i_var = 0; i_var < no_vars_; i_var++)
          {
            g->F_x_flat[id4(i_var, i_z, i_y, i_x)] = F_L[i_var];
          }
        }
        else if (0 <= S_star) // boundary to left of contact discontinuity
        {
          for (i_var = 0; i_var < no_vars_; i_var++)
          {
            g->F_x_flat[id4(i_var, i_z, i_y, i_x)] = F_L[i_var] + 
              S_L * (q_star_L[i_var] - q_L[i_var]);
          }
        }
        else if (0 <= S_R) // boundary to right of contact discontinuity
        {
          for (i_var = 0; i_var < no_vars_; i_var++)
          {
            g->F_x_flat[id4(i_var, i_z, i_y, i_x)] = F_R[i_var] + 
              S_R * (q_star_R[i_var] - q_R[i_var]);
          }
        }
        else
        {
          for (i_var = 0; i_var < no_vars_; i_var++)
          {
            g->F_x_flat[id4(i_var, i_z, i_y, i_x)] = F_R[i_var];
          }
        }
        //if (i_y == 10 && g->i_t == i_t_dump)
        //  fprintf(p_file, "%i, %e, %e, %e, %e, %e, %e, %e\n", i_x, g->F_x[i_x][i_y][i_rho_],
        //    g->F_x[i_x][i_y][i_S_x_], g->F_x[i_x][i_y][i_S_y_], g->F_x[i_x][i_y][i_E_],
        //    p_star, pr_L[i_p_], pr_R[i_p_]);    
      }
    }
  }
      //if (g->i_t == i_t_dump) fclose(p_file);
}
      
////////////////////////////////////////////////////////////////////////////////

#if DIMS_ > 1

void set_F_y(grid *g)
{
  int i_x, i_y, i_z; // loop over grid cells
  
  #if DIMS_ == 3
  #if parallelization_method_ == 1
  #pragma omp parallel for private (i_x, i_y)
  #elif parallelization_method_ == 2
  #pragma acc parallel loop private (i_x, i_y) present(g) collapse(3)
  #endif
  #endif
  #if DIMS_ > 2
  for (i_z = ghosts_; i_z < GRES_z_ - ghosts_ + 1; i_z++)
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
    for (i_y = ghosts_; i_y < GRES_y_ - ghosts_ + 1; i_y++)
    {
      for (i_x = ghosts_; i_x < GRES_x_ - ghosts_ + 1; i_x++)
      {
        int i_var; // loop over state vector entries
        double q_L[no_vars_];
        double pr_L[no_vars_];
        double F_L[no_vars_];
        double q_star_L[no_vars_];
    
        double q_R[no_vars_];
        double pr_R[no_vars_];
        double F_R[no_vars_];
        double q_star_R[no_vars_];
    
        double c_s_L, c_s_R; // sound speeds left and right
        double rho_bar, c_s_bar; // average density and sound speed across boundary
        double p_star; // inner region
      
        double S_L, S_R, S_star; // wave speeds (not momenta)

        // set the left state and flux
        set_L_state(g, i_x, i_y, i_z, y_dir_, q_L, pr_L);

        F_L[i_rho_] = q_L[i_rho_] * pr_L[i_v_y_];
        F_L[i_S_x_] = q_L[i_rho_] * pr_L[i_v_x_] * pr_L[i_v_y_];
        F_L[i_S_y_] = q_L[i_rho_] * pr_L[i_v_y_] * pr_L[i_v_y_] + pr_L[i_p_];
        #if DIMS_ > 2
        F_L[i_S_z_] = q_L[i_rho_] * pr_L[i_v_z_] * pr_L[i_v_y_];
        #endif
        F_L[i_E_] = (q_L[i_E_] + pr_L[i_p_]) * pr_L[i_v_y_];
        F_L[i_tr1_] = q_L[i_tr1_] * pr_L[i_v_y_];

        c_s_L = sqrt(GAMMA_AD_ * pr_L[i_p_] / pr_L[i_rho_]);

        // set the right state and flux
        set_R_state(g, i_x, i_y, i_z, y_dir_, q_R, pr_R);

        F_R[i_rho_] = q_R[i_rho_] * pr_R[i_v_y_];
        F_R[i_S_x_] = q_R[i_rho_] * pr_R[i_v_x_] * pr_R[i_v_y_];
        F_R[i_S_y_] = q_R[i_rho_] * pr_R[i_v_y_] * pr_R[i_v_y_] + pr_R[i_p_];
        #if DIMS_ > 2
        F_R[i_S_z_] = q_R[i_rho_] * pr_R[i_v_z_] * pr_R[i_v_y_];
        #endif
        F_R[i_E_] = (q_R[i_E_] + pr_R[i_p_]) * pr_R[i_v_y_];
        F_R[i_tr1_] = q_R[i_tr1_] * pr_R[i_v_y_];

        c_s_R = sqrt(GAMMA_AD_ * pr_R[i_p_] / pr_R[i_rho_]);

        // Apply the HLLC solver
        rho_bar = 0.5 * (pr_R[i_rho_] + pr_L[i_rho_]);
        c_s_bar = 0.5 * (c_s_L + c_s_R);
        p_star = 0.5  * (pr_L[i_p_] + pr_R[i_p_]) - 
          0.5 * (pr_R[i_v_y_] - pr_L[i_v_y_]) * rho_bar * c_s_bar;
        p_star = fmax(p_star, 0.);

        // HLL left wave speed estimate. Don't confuse wave speed 'S' with
        // momentum 'S'. We're just using the notation from Toro, 3rd ed. page 331
        if (p_star <= pr_L[i_p_])
        {
          S_L = pr_L[i_v_y_] - c_s_L;
        }
        else
        {
          S_L = pr_L[i_v_y_] - c_s_L * 
            sqrt(1. + (GAMMA_AD_ + 1.) / (2. * GAMMA_AD_) * 
              (p_star / pr_L[i_p_] - 1.));
        }
      
        // HLL right wave speed estimate
        if (p_star <= pr_R[i_p_])
        {
          S_R = pr_R[i_v_y_] + c_s_R;
        }
        else
        {
          S_R = pr_R[i_v_y_] + c_s_R *
            sqrt(1. + (GAMMA_AD_ + 1.) / (2. * GAMMA_AD_) * 
              (p_star / pr_R[i_p_] - 1.));
        }

        //----------------------------------------------------------------------
      
        // Begin HLLC-specific estimates (up until here applied to HLL too)
        S_star = (pr_R[i_p_] - pr_L[i_p_] + 
          pr_L[i_rho_] * pr_L[i_v_y_] * (S_L - pr_L[i_v_y_]) -
          pr_R[i_rho_] * pr_R[i_v_y_] * (S_R - pr_R[i_v_y_])) /
          (pr_L[i_rho_] * (S_L - pr_L[i_v_y_]) - 
            pr_R[i_rho_] * (S_R - pr_R[i_v_y_]));

        // starred region state vector estimates, starting with left
        q_star_L[i_rho_] = pr_L[i_rho_] * (S_L - pr_L[i_v_y_]) / (S_L - S_star);

        q_star_L[i_S_x_] = pr_L[i_rho_] * (S_L - pr_L[i_v_y_]) / (S_L - S_star) *
          pr_L[i_v_x_]; // perpendicular momentum just advects, like rho

        q_star_L[i_S_y_] = pr_L[i_rho_] * (S_L - pr_L[i_v_y_]) / (S_L - S_star) *
          S_star;

        #if DIMS_ > 2
        q_star_L[i_S_z_] = pr_L[i_rho_] * (S_L - pr_L[i_v_y_]) / (S_L - S_star) *
          pr_L[i_v_z_]; // perpendicular momentum just advects, like rho
        #endif
        
        q_star_L[i_E_] = pr_L[i_rho_] * (S_L - pr_L[i_v_y_]) / (S_L - S_star) *
          (q_L[i_E_] / pr_L[i_rho_] + (S_star - pr_L[i_v_y_]) * 
            (S_star + pr_L[i_p_] / (pr_L[i_rho_] * (S_L - pr_L[i_v_y_]))));

        q_star_L[i_tr1_] = pr_L[i_tr1_] * (S_L - pr_L[i_v_y_]) / (S_L - S_star);

        // starred region state vector estimates, now the right
        q_star_R[i_rho_] = pr_R[i_rho_] * (S_R - pr_R[i_v_y_]) / (S_R - S_star);

        q_star_R[i_S_x_] = pr_R[i_rho_] * (S_R - pr_R[i_v_y_]) / (S_R - S_star) *
          pr_R[i_v_x_]; // perpendicular momentum just advects, like rho

        q_star_R[i_S_y_] = pr_R[i_rho_] * (S_R - pr_R[i_v_y_]) / (S_R - S_star) *
          S_star;

        #if DIMS_ > 2
        q_star_R[i_S_z_] = pr_R[i_rho_] * (S_R - pr_R[i_v_y_]) / (S_R - S_star) *
          pr_R[i_v_z_]; // perpendicular momentum just advects, like rho
        #endif

        q_star_R[i_E_] = pr_R[i_rho_] * (S_R - pr_R[i_v_y_]) / (S_R - S_star) *
          (q_R[i_E_] / pr_R[i_rho_] + (S_star - pr_R[i_v_y_]) * 
            (S_star + pr_R[i_p_] / (pr_R[i_rho_] * (S_R - pr_R[i_v_y_]))));

        q_star_R[i_tr1_] = pr_R[i_tr1_] * (S_R - pr_R[i_v_y_]) / (S_R - S_star);
      
        // set the flux through the cell boundaries, four possibilities for the
        // for possible regions where the cell boundary can be found
        if (0 <= S_L) // whole wave pattern moved to right of cell boundary
        {
          for (i_var = 0; i_var < no_vars_; i_var++)
          {
            g->F_y_flat[id4(i_var, i_z, i_y, i_x)] = F_L[i_var];
          }
        }
        else if (0 <= S_star) // boundary to left of contact discontinuity
        {
          for (i_var = 0; i_var < no_vars_; i_var++)
          {
            g->F_y_flat[id4(i_var, i_z, i_y, i_x)] = F_L[i_var] + 
              S_L * (q_star_L[i_var] - q_L[i_var]);
          }
        }
        else if (0 <= S_R) // boundary to right of contact discontinuity
        {
          for (i_var = 0; i_var < no_vars_; i_var++)
          {
            g->F_y_flat[id4(i_var, i_z, i_y, i_x)] = F_R[i_var] + 
              S_R * (q_star_R[i_var] - q_R[i_var]);
          }
        }
        else
        {
          for (i_var = 0; i_var < no_vars_; i_var++)
          {
            g->F_y_flat[id4(i_var, i_z, i_y, i_x)] = F_R[i_var];
          }
        }

      }
      //if (i_x == 10 && g->i_t == i_t_dump)
      //  fprintf(p_file, "%i, %e, %e, %e, %e, %e, %e, %e\n", i_y, g->F_y[i_x][i_y][i_rho_], 
      //    g->F_y[i_x][i_y][i_S_x_], g->F_y[i_x][i_y][i_S_y_], g->F_y[i_x][i_y][i_E_],
      //    p_star, pr_L[i_p_], pr_R[i_p_]);
    }
  }
    //if (g->i_t == i_t_dump) fclose(p_file);
}

#endif // DIMS_ > 1

////////////////////////////////////////////////////////////////////////////////

#if DIMS_ > 2

void set_F_z(grid *g)
{
  int i_x, i_y, i_z; // loop over grid cells
  
  #if parallelization_method_ == 1
  #pragma omp parallel for private (i_x, i_y)
  #elif parallelization_method_ == 2
  #pragma acc parallel loop private (i_x, i_y) present(g) collapse(3)
  #endif
  for (i_z = ghosts_; i_z < GRES_z_ - ghosts_ + 1; i_z++)
  {
    for (i_y = ghosts_; i_y < GRES_y_ - ghosts_ + 1; i_y++)
    {
      for (i_x = ghosts_; i_x < GRES_x_ - ghosts_ + 1; i_x++)
      {
        int i_var; // loop over state vector entries
        double q_L[no_vars_];
        double pr_L[no_vars_];
        double F_L[no_vars_];
        double q_star_L[no_vars_];
    
        double q_R[no_vars_];
        double pr_R[no_vars_];
        double F_R[no_vars_];
        double q_star_R[no_vars_];
    
        double c_s_L, c_s_R; // sound speeds left and right
        double rho_bar, c_s_bar; // average density and sound speed across boundary
        double p_star; // inner region
      
        double S_L, S_R, S_star; // wave speeds (not momenta)

        // set the left state and flux
        set_L_state(g, i_x, i_y, i_z, z_dir_, q_L, pr_L);

        F_L[i_rho_] = q_L[i_rho_] * pr_L[i_v_z_];
        F_L[i_S_x_] = q_L[i_rho_] * pr_L[i_v_x_] * pr_L[i_v_z_];
        F_L[i_S_y_] = q_L[i_rho_] * pr_L[i_v_y_] * pr_L[i_v_z_];
        F_L[i_S_z_] = q_L[i_rho_] * pr_L[i_v_z_] * pr_L[i_v_z_] + pr_L[i_p_];
        F_L[i_E_] = (q_L[i_E_] + pr_L[i_p_]) * pr_L[i_v_z_];
        F_L[i_tr1_] = q_L[i_tr1_] * pr_L[i_v_z_];

        c_s_L = sqrt(GAMMA_AD_ * pr_L[i_p_] / pr_L[i_rho_]);

        // set the right state and flux
        set_R_state(g, i_x, i_y, i_z, z_dir_, q_R, pr_R);

        F_R[i_rho_] = q_R[i_rho_] * pr_R[i_v_z_];
        F_R[i_S_x_] = q_R[i_rho_] * pr_R[i_v_x_] * pr_R[i_v_z_];
        F_R[i_S_y_] = q_R[i_rho_] * pr_R[i_v_y_] * pr_R[i_v_z_];
        F_R[i_S_z_] = q_R[i_rho_] * pr_R[i_v_z_] * pr_R[i_v_z_] + pr_R[i_p_];
        F_R[i_E_] = (q_R[i_E_] + pr_R[i_p_]) * pr_R[i_v_z_];
        F_R[i_tr1_] = q_R[i_tr1_] * pr_R[i_v_z_];

        c_s_R = sqrt(GAMMA_AD_ * pr_R[i_p_] / pr_R[i_rho_]);

        // Apply the HLLC solver
        rho_bar = 0.5 * (pr_R[i_rho_] + pr_L[i_rho_]);
        c_s_bar = 0.5 * (c_s_L + c_s_R);
        p_star = 0.5  * (pr_L[i_p_] + pr_R[i_p_]) - 
          0.5 * (pr_R[i_v_z_] - pr_L[i_v_z_]) * rho_bar * c_s_bar;
        p_star = fmax(p_star, 0.);

        // HLL left wave speed estimate. Don't confuse wave speed 'S' with
        // momentum 'S'. We're just using the notation from Toro, 3rd ed. page 331
        if (p_star <= pr_L[i_p_])
        {
          S_L = pr_L[i_v_z_] - c_s_L;
        }
        else
        {
          S_L = pr_L[i_v_z_] - c_s_L * 
            sqrt(1. + (GAMMA_AD_ + 1.) / (2. * GAMMA_AD_) * 
              (p_star / pr_L[i_p_] - 1.));
        }
      
        // HLL right wave speed estimate
        if (p_star <= pr_R[i_p_])
        {
          S_R = pr_R[i_v_z_] + c_s_R;
        }
        else
        {
          S_R = pr_R[i_v_z_] + c_s_R *
            sqrt(1. + (GAMMA_AD_ + 1.) / (2. * GAMMA_AD_) * 
              (p_star / pr_R[i_p_] - 1.));
        }

        //----------------------------------------------------------------------
      
        // Begin HLLC-specific estimates (up until here applied to HLL too)
        S_star = (pr_R[i_p_] - pr_L[i_p_] + 
          pr_L[i_rho_] * pr_L[i_v_z_] * (S_L - pr_L[i_v_z_]) -
          pr_R[i_rho_] * pr_R[i_v_z_] * (S_R - pr_R[i_v_z_])) /
          (pr_L[i_rho_] * (S_L - pr_L[i_v_z_]) - 
            pr_R[i_rho_] * (S_R - pr_R[i_v_z_]));

        // starred region state vector estimates, starting with left
        q_star_L[i_rho_] = pr_L[i_rho_] * (S_L - pr_L[i_v_z_]) / (S_L - S_star);

        q_star_L[i_S_x_] = pr_L[i_rho_] * (S_L - pr_L[i_v_z_]) / (S_L - S_star) *
          pr_L[i_v_x_]; // perpendicular momentum just advects, like rho

        q_star_L[i_S_y_] = pr_L[i_rho_] * (S_L - pr_L[i_v_z_]) / (S_L - S_star) *
          pr_L[i_v_y_]; // perpendicular momentum just advects, like rho

        q_star_L[i_S_z_] = pr_L[i_rho_] * (S_L - pr_L[i_v_z_]) / (S_L - S_star) *
          S_star;

        q_star_L[i_E_] = pr_L[i_rho_] * (S_L - pr_L[i_v_z_]) / (S_L - S_star) *
          (q_L[i_E_] / pr_L[i_rho_] + (S_star - pr_L[i_v_z_]) * 
            (S_star + pr_L[i_p_] / (pr_L[i_rho_] * (S_L - pr_L[i_v_z_]))));

        q_star_L[i_tr1_] = pr_L[i_tr1_] * (S_L - pr_L[i_v_z_]) / (S_L - S_star);

        // starred region state vector estimates, now the right
        q_star_R[i_rho_] = pr_R[i_rho_] * (S_R - pr_R[i_v_z_]) / (S_R - S_star);

        q_star_R[i_S_x_] = pr_R[i_rho_] * (S_R - pr_R[i_v_z_]) / (S_R - S_star) *
          pr_R[i_v_x_]; // perpendicular momentum just advects, like rho

        q_star_R[i_S_y_] = pr_R[i_rho_] * (S_R - pr_R[i_v_z_]) / (S_R - S_star) *
          pr_R[i_v_y_]; // perpendicular momentum just advects, like rho

        q_star_R[i_S_z_] = pr_R[i_rho_] * (S_R - pr_R[i_v_z_]) / (S_R - S_star) *
          S_star;

        q_star_R[i_E_] = pr_R[i_rho_] * (S_R - pr_R[i_v_z_]) / (S_R - S_star) *
          (q_R[i_E_] / pr_R[i_rho_] + (S_star - pr_R[i_v_z_]) * 
            (S_star + pr_R[i_p_] / (pr_R[i_rho_] * (S_R - pr_R[i_v_z_]))));

        q_star_R[i_tr1_] = pr_R[i_tr1_] * (S_R - pr_R[i_v_z_]) / (S_R - S_star);
      
        // set the flux through the cell boundaries, four possibilities for the
        // for possible regions where the cell boundary can be found
        if (0 <= S_L) // whole wave pattern moved to right of cell boundary
        {
          for (i_var = 0; i_var < no_vars_; i_var++)
          {
            g->F_z_flat[id4(i_var, i_z, i_y, i_x)] = F_L[i_var];
          }
        }
        else if (0 <= S_star) // boundary to left of contact discontinuity
        {
          for (i_var = 0; i_var < no_vars_; i_var++)
          {
            g->F_z_flat[id4(i_var, i_z, i_y, i_x)] = F_L[i_var] + 
              S_L * (q_star_L[i_var] - q_L[i_var]);
          }
        }
        else if (0 <= S_R) // boundary to right of contact discontinuity
        {
          for (i_var = 0; i_var < no_vars_; i_var++)
          {
            g->F_z_flat[id4(i_var, i_z, i_y, i_x)] = F_R[i_var] + 
              S_R * (q_star_R[i_var] - q_R[i_var]);
          }
        }
        else
        {
          for (i_var = 0; i_var < no_vars_; i_var++)
          {
            g->F_z_flat[id4(i_var, i_z, i_y, i_x)] = F_R[i_var];
          }
        }

      }
      //if (i_x == 10 && g->i_t == i_t_dump)
      //  fprintf(p_file, "%i, %e, %e, %e, %e, %e, %e, %e\n", i_y, g->F_y[i_x][i_y][i_rho_], 
      //    g->F_y[i_x][i_y][i_S_x_], g->F_y[i_x][i_y][i_S_y_], g->F_y[i_x][i_y][i_E_],
      //    p_star, pr_L[i_p_], pr_R[i_p_]);
      
    }
  }
    //if (g->i_t == i_t_dump) fclose(p_file);
}

#endif // DIMS_ > 2


