// IO.c

#include "IO.h"

void print_state(grid *g)
{
  int i_x;
  double dx, x;
  dx = (g->x_max - g->x_min) / (RES_x_);

  #if DIMS_ > 1
  int i_y;
  double dy, y;
  dy = (g->y_max - g->y_min) / (RES_y_);
  #endif
  
  #if DIMS_ > 2
  int i_z;
  double dz, z;
  dz = (g->z_max - g->z_min) / (RES_z_);
  #endif

  // for GPU parallelization, retrieve info from the device and update the host,
  // pulling the state from layer 0
  #if parallelization_method_ == 2
  #pragma acc update self(g->q_flat[0:no_vars_ * GRES_z_ * GRES_y_ * GRES_x_])
  #pragma acc update self(g->pr_flat[0:no_vars_ * GRES_z_ * GRES_y_ * GRES_x_])
  #endif

  //----------------------------------------------------------------------------

  #if DIMS_ == 1

  for (i_x = ghosts_; i_x < RES_x_ + ghosts_; i_x++)
  {
    x = g->x_min + (i_x - ghosts_ + 0.5) * dx;
    printf("%1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e\n",
      x, g->q[0][i_rho_][0][0][i_x], 
      g->q[0][i_S_x_][0][0][i_x],
      g->q[0][i_E_][0][0][i_x], 
      g->pr[i_v_x_][0][0][i_x],
      g->pr[i_p_][0][0][i_x],
      g->q[0][i_tr1_][0][0][i_x]);
  }
  
  #endif

  //----------------------------------------------------------------------------

  #if DIMS_ == 2

  for (i_x = ghosts_; i_x < RES_x_ + ghosts_; i_x++)
  {
    x = g->x_min + (i_x - ghosts_ + 0.5) * dx;
    for (i_y = ghosts_; i_y < RES_y_ + ghosts_; i_y++)
    {
      y = g->y_min + (i_y - ghosts_ + 0.5) * dy;
      printf("%1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e,"
        " %1.3e\n",
        x, y, g->q[0][i_rho_][0][i_y][i_x], 
        g->q[0][i_S_x_][0][i_y][i_x],
        g->q[0][i_S_y_][0][i_y][i_x], 
        g->q[0][i_E_][0][i_y][i_x], 
        g->pr[i_v_x_][0][i_y][i_x],
        g->pr[i_v_y_][0][i_y][i_x], 
        g->pr[i_p_][0][i_y][i_x],
        g->q[0][i_tr1_][0][i_y][i_x]);
    }
  }
  
  #endif
  
  //----------------------------------------------------------------------------

  #if DIMS_ == 3

  for (i_x = ghosts_; i_x < RES_x_ + ghosts_; i_x++)
  {
    x = g->x_min + (i_x - ghosts_ + 0.5) * dx;
    for (i_y = ghosts_; i_y < RES_y_ + ghosts_; i_y++)
    {
      y = g->y_min + (i_y - ghosts_ + 0.5) * dy;
      for (i_z = ghosts_; i_z < RES_z_ + ghosts_; i_z++)
      {
        z = g->z_min + (i_z - ghosts_ + 0.5) * dz;
        printf("%1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e,"
          " %1.3e, %1.3e, %1.3e, %1.3e\n",
          x, y, z, g->q[0][i_rho_][i_z][i_y][i_x], 
          g->q[0][i_S_x_][i_z][i_y][i_x],
          g->q[0][i_S_y_][i_z][i_y][i_x], 
          g->q[0][i_S_z_][i_z][i_y][i_x],
          g->q[0][i_E_][i_z][i_y][i_x], 
          g->pr[i_v_x_][i_z][i_y][i_x],
          g->pr[i_v_y_][i_z][i_y][i_x], 
          g->pr[i_v_z_][i_z][i_y][i_x], 
          g->pr[i_p_][i_z][i_y][i_x],
          g->q[0][i_tr1_][i_z][i_y][i_x]);
      }
    }
  }
  
  #endif
}

////////////////////////////////////////////////////////////////////////////////

void print_state_ghosts_(grid *g)
{
  int i_x;
  double dx, x;
  dx = (g->x_max - g->x_min) / (RES_x_);

  #if DIMS_ > 1
  int i_y;
  double dy, y;
  dy = (g->y_max - g->y_min) / (RES_y_);
  #endif
  
  #if DIMS_ > 2
  int i_z;
  double dz, z;
  dz = (g->z_max - g->z_min) / (RES_z_);
  #endif

  // for GPU parallelization, retrieve info from the device and update the host,
  // pulling the state from layer 0
  #if parallelization_method_ == 2
  #pragma acc update self(g->q_flat[0:no_vars_ * GRES_z_ * GRES_y_ * GRES_x_])
  #pragma acc update self(g->pr_flat[0:no_vars_ * GRES_z_ * GRES_y_ * GRES_x_])
  #endif

  //----------------------------------------------------------------------------

  #if DIMS_ == 1

  for (i_x = 0; i_x < GRES_x_; i_x++)
  {
    x = g->x_min + (i_x - ghosts_ + 0.5) * dx;
    printf("%1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e\n",
      x, g->q[0][i_rho_][0][0][i_x], 
      g->q[0][i_S_x_][0][0][i_x],
      g->q[0][i_E_][0][0][i_x], 
      g->pr[i_v_x_][0][0][i_x],
      g->pr[i_p_][0][0][i_x],
      g->q[0][i_tr1_][0][0][i_x]);
  }
  
  #endif

  //----------------------------------------------------------------------------

  #if DIMS_ == 2

  for (i_x = 0; i_x < GRES_x_; i_x++)
  {
    x = g->x_min + (i_x - ghosts_ + 0.5) * dx;
    for (i_y = 0; i_y < GRES_y_; i_y++)
    {
      y = g->y_min + (i_y - ghosts_ + 0.5) * dy;
      printf("%1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e,"
        " %1.3e\n",
        x,
        y, 
        g->q[0][i_rho_][0][i_y][i_x], 
        g->q[0][i_S_x_][0][i_y][i_x],
        g->q[0][i_S_y_][0][i_y][i_x], 
        g->q[0][i_E_][0][i_y][i_x], 
        g->pr[i_v_x_][0][i_y][i_x],
        g->pr[i_v_y_][0][i_y][i_x], 
        g->pr[i_p_][0][i_y][i_x],
        g->q[0][i_tr1_][0][i_y][i_x]);
    }
  }
  
  #endif
  
  //----------------------------------------------------------------------------

  #if DIMS_ == 3

  for (i_x = 0; i_x < GRES_x_; i_x++)
  {
    x = g->x_min + (i_x - ghosts_ + 0.5) * dx;
    for (i_y = 0; i_y < GRES_y_; i_y++)
    {
      y = g->y_min + (i_y - ghosts_ + 0.5) * dy;
      for (i_z = 0; i_z < GRES_z_; i_z++)
      {
        z = g->z_min + (i_z - ghosts_ + 0.5) * dz;
        printf("%1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e,"
          " %1.3e, %1.3e, %1.3e, %1.3e\n",
          x, y, z, g->q[0][i_rho_][i_z][i_y][i_x], 
          g->q[0][i_S_x_][i_z][i_y][i_x],
          g->q[0][i_S_y_][i_z][i_y][i_x], 
          g->q[0][i_S_z_][i_z][i_y][i_x],
          g->q[0][i_E_][i_z][i_y][i_x], 
          g->pr[i_v_x_][i_z][i_y][i_x],
          g->pr[i_v_y_][i_z][i_y][i_x], 
          g->pr[i_v_z_][i_z][i_y][i_x], 
          g->pr[i_p_][i_z][i_y][i_x],
          g->q[0][i_tr1_][i_z][i_y][i_x]);
      }
    }
  }
  
  #endif
}


////////////////////////////////////////////////////////////////////////////////

void fprint_state(const char* filename, grid *g)
{
  FILE *p_file;
  p_file = fopen(filename, "w");
  
  // print header info
  #if DIMS_ < 3 // anticipate plotting using python matplotlib
  fprintf(p_file, "# running set-up labeled: %s\n", g->title);
  fprintf(p_file, "# Adiabatic exponent set to: %f\n", GAMMA_AD_);
  fprintf(p_file, "# current grid time: %e\n", g->t);
  fprintf(p_file, "# X resolution: %d\n", RES_x_);
  #endif


  int i_x;
  double dx, x;
  dx = (g->x_max - g->x_min) / (RES_x_);

  #if DIMS_ > 1
  int i_y;
  double dy, y;
  dy = (g->y_max - g->y_min) / (RES_y_);

  #if DIMS_ < 3
  fprintf(p_file, "# Y resolution: %d\n", RES_y_);
  #endif
  #endif
  
  #if DIMS_ > 2
  int i_z;
  double dz, z;
  dz = (g->z_max - g->z_min) / (RES_z_);

  //fprintf(p_file, "# Z resolution: %d\n", RES_z_);
  #endif

  // for GPU parallelization, retrieve info from the device and update the host,
  // pulling the state from layer 0
  #if parallelization_method_ == 2
  #pragma acc update self(g->q_flat[0:no_vars_ * GRES_z_ * GRES_y_ * GRES_x_])
  #pragma acc update self(g->pr_flat[0:no_vars_ * GRES_z_ * GRES_y_ * GRES_x_])
  #endif
 
  //----------------------------------------------------------------------------

  #if DIMS_ == 1

  for (i_x = ghosts_; i_x < RES_x_ + ghosts_; i_x++)
  {
    x = g->x_min + (i_x - ghosts_ + 0.5) * dx;
    fprintf(p_file, "%1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e\n",
      x, 
      g->q[0][i_rho_][0][0][i_x], 
      g->q[0][i_S_x_][0][0][i_x],
      g->q[0][i_E_][0][0][i_x], 
      g->pr[i_v_x_][0][0][i_x],
      g->pr[i_p_][0][0][i_x],
      g->q[0][i_tr1_][0][0][i_x]);
  }
  
  #endif

  //----------------------------------------------------------------------------

  #if DIMS_ == 2

  for (i_x = ghosts_; i_x < RES_x_ + ghosts_; i_x++)
  {
    x = g->x_min + (i_x - ghosts_ + 0.5) * dx;
    for (i_y = ghosts_; i_y < RES_y_ + ghosts_; i_y++)
    {
      y = g->y_min + (i_y - ghosts_ + 0.5) * dy;
      fprintf(p_file, "%1.5f, %1.5f, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e, "
        "%1.3e, %1.3e\n",
        x, y, g->q[0][i_rho_][0][i_y][i_x], 
        g->q[0][i_S_x_][0][i_y][i_x],
        g->q[0][i_S_y_][0][i_y][i_x], 
        g->q[0][i_E_][0][i_y][i_x], 
        g->pr[i_v_x_][0][i_y][i_x],
        g->pr[i_v_y_][0][i_y][i_x], 
        g->pr[i_p_][0][i_y][i_x],
        g->q[0][i_tr1_][0][i_y][i_x]);
    }
  }
  
  #endif
  
  //----------------------------------------------------------------------------

  #if DIMS_ == 3


  fprintf(p_file, "x,y,z,rho,Sx,Sy,Sz,E,vx,vy,vz,p,tr1\n"); // trying to get paraview interaction going

  for (i_x = ghosts_; i_x < RES_x_ + ghosts_; i_x++)
  {
    x = g->x_min + (i_x - ghosts_ + 0.5) * dx;
    for (i_y = ghosts_; i_y < RES_y_ + ghosts_; i_y++)
    {
      y = g->y_min + (i_y - ghosts_ + 0.5) * dy;
      for (i_z = ghosts_; i_z < RES_z_ + ghosts_; i_z++)
      {
        z = g->z_min + (i_z - ghosts_ + 0.5) * dz;
        //fprintf(p_file, "%1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e,"
        //  " %1.3e, %1.3e, %1.3e, %1.3e, %1.3e, %1.3e\n",
        fprintf(p_file, "%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,"
          "%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f\n",
          x, y, z, g->q[0][i_rho_][i_z][i_y][i_x], 
          g->q[0][i_S_x_][i_z][i_y][i_x],
          g->q[0][i_S_y_][i_z][i_y][i_x], 
          g->q[0][i_S_z_][i_z][i_y][i_x],
          g->q[0][i_E_][i_z][i_y][i_x], 
          g->pr[i_v_x_][i_z][i_y][i_x],
          g->pr[i_v_y_][i_z][i_y][i_x], 
          g->pr[i_v_z_][i_z][i_y][i_x], 
          g->pr[i_p_][i_z][i_y][i_x],
          g->q[0][i_tr1_][i_z][i_y][i_x]);
      }
    }
  }
  
  #endif
  
  fclose(p_file);
}
