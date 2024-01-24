// reconstruct.c

#include "reconstruct.h"

#if parallelization_method_ == 2
#pragma acc routine seq
#endif
void set_L_state(grid *g, int i_x, int i_y, int i_z, char dir, double *q_L, 
  double *pr_L)
{
  double a, b, c, s;
  double sa, sb, sc; // signs of a, b, c
  int i_var;

  if (dir == x_dir_)
  {
    double dx = (g->x_max - g->x_min) / (RES_x_);
    for (i_var = 0; i_var < no_vars_; i_var++)
    {
      //printf("i_var = %i, i_x = %i, i_y = %i, i_z = %i\n", i_var, i_x, i_y, i_z); fflush(stdout);
      a = (g->pr_flat[id4(i_var, i_z, i_y, i_x-1)] - g->pr_flat[id4(i_var, i_z, i_y, i_x-2)]) / dx;
      b = (g->pr_flat[id4(i_var, i_z, i_y, i_x)] - g->pr_flat[id4(i_var, i_z, i_y, i_x-1)]) / dx;
      c = (g->pr_flat[id4(i_var, i_z, i_y, i_x)] - g->pr_flat[id4(i_var, i_z, i_y, i_x-2)]) / (2.*dx);
      
      sa = copysign(1., a);
      sb = copysign(1., b);
      sc = copysign(1., c);
      //printf("a = %e, sa = %f, b = %e, sb = %f, c = %e, sc = %f\n",
      //  a, sa, b, sb, c, sc); fflush(stdout);
      s = 0.25 * sa * (sa + sb) * (sa + sc) * 
        fmin(fmin(fabs(a), fabs(b)), fabs(c));
      
      #if disable_PLM_
      s = 0;
      #endif
      pr_L[i_var] = g->pr_flat[id4(i_var, i_z, i_y, i_x-1)] + 0.5 * dx * s;
    }
  }

  if (dir == y_dir_)
  {
    double dy = (g->y_max - g->y_min) / (RES_y_);
    for (i_var = 0; i_var < no_vars_; i_var++)
    {
      //printf("i_var = %i, i_x = %i, i_y = %i, i_z = %i\n", i_var, i_x, i_y, i_z); fflush(stdout);
      a = (g->pr_flat[id4(i_var, i_z, i_y-1, i_x)] - g->pr_flat[id4(i_var, i_z, i_y-2, i_x)]) / dy;
      b = (g->pr_flat[id4(i_var, i_z, i_y, i_x)] - g->pr_flat[id4(i_var, i_z, i_y-1, i_x)]) / dy;
      c = (g->pr_flat[id4(i_var, i_z, i_y, i_x)] - g->pr_flat[id4(i_var, i_z, i_y-2, i_x)]) / (2.*dy);
      
      sa = copysign(1., a);
      sb = copysign(1., b);
      sc = copysign(1., c);
      s = 0.25 * sa * (sa + sb) * (sa + sc) * 
        fmin(fmin(fabs(a), fabs(b)), fabs(c));
      
      #if disable_PLM_
      s = 0;
      #endif
      pr_L[i_var] = g->pr_flat[id4(i_var, i_z, i_y-1, i_x)] + 0.5 * dy * s;
    }
  }

  if (dir == z_dir_)
  {
    double dz = (g->z_max - g->z_min) / (RES_z_);
    for (i_var = 0; i_var < no_vars_; i_var++)
    {
      a = (g->pr_flat[id4(i_var, i_z-1, i_y, i_x)] - g->pr_flat[id4(i_var, i_z-2, i_y, i_x)]) / dz;
      b = (g->pr_flat[id4(i_var, i_z, i_y, i_x)] - g->pr_flat[id4(i_var, i_z-1, i_y, i_x)]) / dz;
      c = (g->pr_flat[id4(i_var, i_z, i_y, i_x)] - g->pr_flat[id4(i_var, i_z-2, i_y, i_x)]) / (2.*dz);
      
      sa = copysign(1., a);
      sb = copysign(1., b);
      sc = copysign(1., c);
      s = 0.25 * sa * (sa + sb) * (sa + sc) * 
        fmin(fmin(fabs(a), fabs(b)), fabs(c));
      
      #if disable_PLM_
      s = 0;
      #endif
      pr_L[i_var] = g->pr_flat[id4(i_var, i_z-1, i_y, i_x)] + 0.5 * dz * s;
    }
  }
  
  // apply floors
  pr_L[i_p_] = fmax(pr_L[i_p_], p_floor_);
  pr_L[i_rho_] = fmax(pr_L[i_rho_], rho_floor_);
  
  // set local conserved variables
  prim2cons_local(pr_L, q_L);
}

////////////////////////////////////////////////////////////////////////////////

#if parallelization_method_ == 2
#pragma acc routine seq
#endif
void set_R_state(grid *g, int i_x, int i_y, int i_z, char dir, double *q_R, 
  double *pr_R)
{
  double a, b, c, s;
  double sa, sb, sc; // signs of a, b, c
  int i_var;
  double dx, dy, dz;

  dx = (g->x_max - g->x_min) / (RES_x_);
  dy = (g->y_max - g->y_min) / (RES_y_);
  dz = (g->z_max - g->z_min) / (RES_z_);
 
  if (dir == x_dir_)
  {
    for (i_var = 0; i_var < no_vars_; i_var++)
    {
      a = (g->pr_flat[id4(i_var, i_z, i_y, i_x)] - g->pr_flat[id4(i_var, i_z, i_y, i_x-1)]) / dx;
      b = (g->pr_flat[id4(i_var, i_z, i_y, i_x+1)] - g->pr_flat[id4(i_var, i_z, i_y, i_x)]) / dx;
      c = (g->pr_flat[id4(i_var, i_z, i_y, i_x+1)] - g->pr_flat[id4(i_var, i_z, i_y, i_x-1)]) / (2.*dx);
      
      sa = copysign(1., a);
      sb = copysign(1., b);
      sc = copysign(1., c);
      s = 0.25 * sa * (sa + sb) * (sa + sc) * 
        fmin(fmin(fabs(a), fabs(b)), fabs(c));
      
      #if disable_PLM_
      s = 0;
      #endif
      pr_R[i_var] = g->pr_flat[id4(i_var, i_z, i_y, i_x)] - 0.5 * dx * s;
    }
  }
  
  if (dir == y_dir_)
  {
    for (i_var = 0; i_var < no_vars_; i_var++)
    {
      a = (g->pr_flat[id4(i_var, i_z, i_y, i_x)] - g->pr_flat[id4(i_var, i_z, i_y-1, i_x)]) / dy;
      b = (g->pr_flat[id4(i_var, i_z, i_y+1, i_x)] - g->pr_flat[id4(i_var, i_z, i_y, i_x)]) / dy;
      c = (g->pr_flat[id4(i_var, i_z, i_y+1, i_x)] - g->pr_flat[id4(i_var, i_z, i_y-1, i_x)]) / (2.*dy);
      
      sa = copysign(1., a);
      sb = copysign(1., b);
      sc = copysign(1., c);
      s = 0.25 * sa * (sa + sb) * (sa + sc) * 
        fmin(fmin(fabs(a), fabs(b)), fabs(c));
      
      #if disable_PLM_
      s = 0;
      #endif
     pr_R[i_var] = g->pr_flat[id4(i_var, i_z, i_y, i_x)] - 0.5 * dy * s;
    }
  }
  
  if (dir == z_dir_)
  {
    for (i_var = 0; i_var < no_vars_; i_var++)
    {
      a = (g->pr_flat[id4(i_var, i_z, i_y, i_x)] - g->pr_flat[id4(i_var, i_z-1, i_y, i_x)]) / dz;
      b = (g->pr_flat[id4(i_var, i_z+1, i_y, i_x)] - g->pr_flat[id4(i_var, i_z, i_y, i_x)]) / dz;
      c = (g->pr_flat[id4(i_var, i_z+1, i_y, i_x)] - g->pr_flat[id4(i_var, i_z-1, i_y, i_x)]) / (2.*dz);
      
      //printf("a = %e, i_x = %i, i_y = %i, i_z = %i, i_var = %i\n", a, i_x, i_y, i_z, i_var); fflush(stdout);
      
      sa = copysign(1., a);
      sb = copysign(1., b);
      sc = copysign(1., c);
      s = 0.25 * sa * (sa + sb) * (sa + sc) * 
        fmin(fmin(fabs(a), fabs(b)), fabs(c));
      
      #if disable_PLM_
      s = 0;
      #endif
     pr_R[i_var] = g->pr_flat[id4(i_var, i_z, i_y, i_x)] - 0.5 * dz * s;
    }
  }
  
  // apply floors tot avoid NaN
  pr_R[i_p_] = fmax(pr_R[i_p_], p_floor_);
  pr_R[i_rho_] = fmax(pr_R[i_rho_], rho_floor_);

  // set local primitive variables
  prim2cons_local(pr_R, q_R);
}
