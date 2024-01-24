// consprim routines to go from conserved to primitive variables and back

#include "consprim.h"

////////////////////////////////////////////////////////////////////////////////

void cons2prim(grid *g, int layer)
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
  for (i_z = 0; i_z < GRES_z_; i_z++)
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
    for (i_y = 0; i_y < GRES_y_; i_y++)
    #else
    i_y = 0;
    #endif
    {
      #if DIMS_ == 1 && parallelization_method_ == 2
      #pragma acc parallel loop private (i_x) present(g)
      #endif
      for (i_x = 0; i_x < GRES_x_; i_x++)
      {
        double vsqr; // velocity squared

        int id_rho = i_rho_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;

        int id_S_x = i_S_x_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;
        int id_v_x = i_v_x_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;
        #if DIMS_ > 1
        int id_S_y = i_S_y_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;
        int id_v_y = i_v_y_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;
        #endif
        #if DIMS_ > 2
        int id_S_z = i_S_z_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;
        int id_v_z = i_v_z_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;
        #endif

        int id_E = i_E_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;
        int id_p = i_p_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;
        int id_tr1 = i_tr1_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;

        int offset_layer = layer * no_vars_ * GRES_z_ * GRES_y_ * GRES_x_;

        g->pr_flat[id_rho] = g->q_flat[offset_layer + id_rho];
        g->pr_flat[id_v_x] = g->q_flat[offset_layer + id_S_x] / 
          g->q_flat[offset_layer + id_rho];
        vsqr = g->pr_flat[id_v_x] * g->pr_flat[id_v_x];
        #if DIMS_ > 1
        g->pr_flat[id_v_y] = g->q_flat[offset_layer + id_S_y] / 
          g->q_flat[offset_layer + id_rho];
        vsqr += g->pr_flat[id_v_y] * g->pr_flat[id_v_y];
        #endif
        #if DIMS_ > 2
        g->pr_flat[id_v_z] = g->q_flat[offset_layer + id_S_z] / 
          g->q_flat[offset_layer + id_rho];
        vsqr += g->pr_flat[id_v_z] * g->pr_flat[id_v_z];
        #endif
        g->pr_flat[id_p] = (g->q_flat[offset_layer + id_E] - 
          0.5 * g->q_flat[offset_layer + id_rho] * vsqr) * (GAMMA_AD_ - 1.);

        g->pr_flat[id_tr1] = g->q_flat[offset_layer + id_tr1];
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

void prim2cons(grid *g)
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
    #if DIMS_ > 1
    for (i_y = ghosts_; i_y < GRES_y_ - ghosts_; i_y++)
    #else
    i_y = 0;
    #endif
    {
      #if DIMS_ == 1 && parallelization_method_ == 2
      #pragma acc parallel loop private (i_x) present(g)
      #endif
      for (i_x = ghosts_; i_x < GRES_x_ - ghosts_; i_x++)
      {
        double vsqr; // velocity squared
        
        int id_rho = i_rho_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;

        int id_S_x = i_S_x_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;
        int id_v_x = i_v_x_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;
        #if DIMS_ > 1
        int id_S_y = i_S_y_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;
        int id_v_y = i_v_y_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;
        #endif
        #if DIMS_ > 2
        int id_S_z = i_S_z_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;
        int id_v_z = i_v_z_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;
        #endif

        int id_E = i_E_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;
        int id_p = i_p_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;
        int id_tr1 = i_tr1_ * GRES_z_ * GRES_y_ * GRES_x_ +
          i_z * GRES_y_ * GRES_x_ +
          i_y * GRES_x_ +
          i_x;

        g->q_flat[id_rho] = g->pr_flat[id_rho];
        
        g->q_flat[id_S_x] = g->q_flat[id_rho] * g->pr_flat[id_v_x];

        vsqr = g->pr_flat[id_v_x] * g->pr_flat[id_v_x];
        #if DIMS_ > 1
        g->q_flat[id_S_y] = g->q_flat[id_rho] * g->pr_flat[id_v_y];
        vsqr += g->pr_flat[id_v_y] * g->pr_flat[id_v_y];
        #endif
        #if DIMS_ > 2
        g->q_flat[id_S_z] = g->q_flat[id_rho] * g->pr_flat[id_v_z];
        vsqr += g->pr_flat[id_v_z] * g->pr_flat[id_v_z];
        #endif
      
        g->q_flat[id_E] = 0.5 * g->pr_flat[id_rho] * vsqr +
          g->pr_flat[id_p] / (GAMMA_AD_ - 1.);
  
        g->q_flat[id_tr1] = g->pr_flat[id_tr1];
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

#if parallelization_method_ == 2
#pragma acc routine seq
#endif
void cons2prim_local(double *q, double *pr)
{
  double vsqr;
  
  pr[i_rho_] = q[i_rho_];
  pr[i_v_x_] = q[i_S_x_] / q[i_rho_];
  vsqr = pr[i_v_x_] * pr[i_v_x_];
  #if DIMS_ > 1
  pr[i_v_y_] = q[i_S_y_] / q[i_rho_];
  vsqr += pr[i_v_y_] * pr[i_v_y_];
  #endif
  #if DIMS_ > 2
  pr[i_v_z_] = q[i_S_z_] / q[i_rho_];
  vsqr += pr[i_v_z_] * pr[i_v_z_];
  #endif

  pr[i_p_] = (q[i_E_] - 0.5 * q[i_rho_] * vsqr) * (GAMMA_AD_ - 1.);

  pr[i_tr1_] = q[i_tr1_];
}

////////////////////////////////////////////////////////////////////////////////

#if parallelization_method_ == 2
#pragma acc routine seq
#endif
void prim2cons_local(double *pr, double *q)
{
  double vsqr;
  
  q[i_rho_] = pr[i_rho_];
  q[i_S_x_] = pr[i_rho_] * pr[i_v_x_];
  vsqr = pr[i_v_x_] * pr[i_v_x_];
  #if DIMS_ > 1
  q[i_S_y_] = pr[i_rho_] * pr[i_v_y_];
  vsqr += pr[i_v_y_] * pr[i_v_y_];
  #endif
  #if DIMS_ > 2
  q[i_S_z_] = pr[i_rho_] * pr[i_v_z_];
  vsqr += pr[i_v_z_] * pr[i_v_z_];
  #endif

  q[i_E_] = 0.5 * pr[i_rho_] * vsqr + pr[i_p_] / (GAMMA_AD_ - 1.);

  q[i_tr1_] = pr[i_tr1_];
}





