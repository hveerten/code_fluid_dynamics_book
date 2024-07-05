# python-hydro
#
# An implementation of a finite volume 1D hydro solver in python, done as simple
# as possible to demonstrate the CFD concepts rather than python. 
#
# This version uses the HLL method, the Piecewise Linear Method and a 
# third order Runge-Kutta time step.

import numpy as np

# resolution settings and other program settings
RES = 200 # set the numerical resolution, excluding ghost cells
no_ghosts = 2 # number of ghost cells, should be 2 for Piecewise Linear Method
itmax = 100000 # maximum number of iterations, use negative number to ignore
plot_output = True # set to True if you wish a figure to be shown at completion

# physics settings
gamma = 1.4 # adiabatic exponent, assuming adiabatic exponent EOS
x0 = 0. # x-coordinate left boundary grid (ghost cells lie beyond this)
x1 = 1. # x-coordinate right boundary grid (ghost cells lie beyond this)
t1 = 0.2 # maximum running time

# Initialize computed grid global variables
t = 0. # current time
dt = 0. # current time step size
iterations = 0 # total number of iterations
finished = False

# derived quantities
i0 = no_ghosts # first non-ghost cell entry number
i1 = RES + no_ghosts # last non-ghost cell entry number + 1
ig0 = 0 # first entry number entire array
ig1 = RES + 2 * no_ghosts # last entry number entire array + 1
dx = (x1 - x0) / RES # size of a cell, assuming regular grid

# some default masks to select subsets of array entries
grid_entries = np.empty(RES + 2 * no_ghosts, dtype=bool)
grid_entries[0:no_ghosts] = False
grid_entries[no_ghosts:no_ghosts+RES] = True
grid_entries[no_ghosts+RES:no_ghosts*2+RES] = False

flux_entries = np.empty(RES + 2 * no_ghosts, dtype = bool)
flux_entries[0:no_ghosts] = False
flux_entries[no_ghosts:no_ghosts+RES+1] = True
flux_entries[no_ghosts+RES+1:no_ghosts*2+RES] = False

#-------------------------------------------------------------------------------

# initialize the arrays
rho = np.empty((RES + 2 * no_ghosts, 3)) # conserved variable density
rhov = np.empty((RES + 2 * no_ghosts, 3)) # conserved variable momentum density
E = np.empty((RES + 2 * no_ghosts, 3)) # conserved variable energy density

p = np.empty(RES + 2 * no_ghosts)   # primitive variable pressure
v = np.empty(RES + 2 * no_ghosts)   # primitive variable velocity

Frho = np.empty(RES + 2 * no_ghosts) # density flux left cell boundary
Frhov = np.empty(RES + 2 * no_ghosts) # momentum flux left cell boundary
FE = np.empty(RES + 2 * no_ghosts) # energy flux left cell boundary

x = np.empty(RES + 2 * no_ghosts) # x-coordinate left boundary of cell
for i in range(ig1):
  x[i] = x0 + (i - no_ghosts) * dx

# We'll be greedy with our memory needs to make for a simple code (it is 1D, so
# we can easily afford to be). Again, the focus is a clear demonstration of the
# concepts, not an optimized code. Therefore, we just declare extra arrays for
# various auxiliary quantities. quantities with label 'L' are defined just
# to the left of the left cell boundary of cell i. Quantities with label 'R' are
# defined just to the right of the left cell boundary of cell i.
dt_local = np.empty(RES + 2 * no_ghosts) # local CFL condition

rhoL = np.empty(RES + 2 * no_ghosts) # conserved variable density
rhovL = np.empty(RES + 2 * no_ghosts) # conserved variable momentum density
EL = np.empty(RES + 2 * no_ghosts) # conserved variable energy density

pL = np.empty(RES + 2 * no_ghosts)   # primitive variable pressure
vL = np.empty(RES + 2 * no_ghosts)   # primitive variable velocity

csL = np.empty(RES + 2 * no_ghosts) # sound speed left

rhoR = np.empty(RES + 2 * no_ghosts) # conserved variable density
rhovR = np.empty(RES + 2 * no_ghosts) # conserved variable momentum density
ER = np.empty(RES + 2 * no_ghosts) # conserved variable energy density

pR = np.empty(RES + 2 * no_ghosts)   # primitive variable pressure
vR = np.empty(RES + 2 * no_ghosts)   # primitive variable velocity

csR = np.empty(RES + 2 * no_ghosts) # sound speed right

FrhoL = np.empty(RES + 2 * no_ghosts) # density flux left cell boundary
FrhovL = np.empty(RES + 2 * no_ghosts) # momentum flux left cell boundary
FEL = np.empty(RES + 2 * no_ghosts) # energy flux left cell boundary

FrhoR = np.empty(RES + 2 * no_ghosts) # density flux left cell boundary
FrhovR = np.empty(RES + 2 * no_ghosts) # momentum flux left cell boundary
FER = np.empty(RES + 2 * no_ghosts) # energy flux left cell boundary

pstar = np.empty(RES + 2 * no_ghosts) # p-star from HLL method
SL = np.empty(RES + 2 * no_ghosts) # left wave speed left cell boundary
SR = np.empty(RES + 2 * no_ghosts) # right wave speed left cell boundary

sa = np.empty(RES + 2 * no_ghosts) # 'raw' slopes, no minmod applied
sb = np.empty(RES + 2 * no_ghosts) # 'raw' slopes, no minmod applied
s = np.empty(RES + 2 * no_ghosts) # slopes

#-------------------------------------------------------------------------------
 
def prim2cons(RK = 0):
  # compute conserved quantities rho, S = rho v, E based on primitive rho, p, v
  # (rho is both, so does not need separate computing). 
  # Only acts on non-ghost cells.
  rhov[i0:i1, RK] = rho[i0:i1, RK] * v[i0:i1]
  E[i0:i1, RK] = 0.5 * v[i0:i1] * rhov[i0:i1, RK] + p[i0:i1] / (gamma - 1.)
  
def cons2prim(RK = 0):
  # compute primitive quantities rho, p, v based on conserved rho, S, E
  # (rho is both, so does not need separate computing).
  # Acts on ghost cells as well
  v[ig0:ig1] = rhov[ig0:ig1, RK] / rho[ig0:ig1, RK]
  p[ig0:ig1] = ((gamma - 1.) * 
    (E[ig0:ig1, RK] - .5 * rhov[ig0:ig1, RK] * v[ig0:ig1])) 

def set_ghosts(RK = 0):
  # set the conserved quantity values of the ghost cells
  for i in range(no_ghosts):
 
    # inflow/outflow BC left side
    rho[i, RK] = rho[no_ghosts, RK]
    rhov[i, RK] = rhov[no_ghosts, RK]
    E[i, RK] = E[no_ghosts, RK]
    
    # inflow/outflow BC right side
    rho[i1 + i, RK] = rho[i1 - 1, RK]
    rhov[i1 + i, RK] = rhov[i1 - 1, RK]
    E[i1 + i, RK] = E[i1 - 1, RK]

def set_LR_states(RK = 0):
  # Set the left and right states based on the Piecewise Linear Method and 
  # minmod slope modifier. We'll apply PLM to rho, p and v. The quantities
  # E and rhov are then reproduced from these, rather than by interpolation
  # between their values at adjacent cells. This ensures that the fluid state
  # variables are consistent
  
  # density states
  sa[0] = 0
  sa[i0-1:i1+2] = (rho[i0-1:i1+2, RK] - rho[i0-2:i1+1, RK]) / dx
  sb[i0-1:i1+1] = sa[i0:i1+2]
  
  s[:] = 0 # by default
  
  entries = (sa < 0) & (sb < 0)
  s[entries] = np.maximum(sa[entries], sb[entries])
  
  entries = (sa > 0) & (sb > 0)
  s[entries] = np.minimum(sa[entries], sb[entries])
  
  rhoL[i0:i1+1] = rho[i0-1:i1, RK] + 0.5 * dx * s[i0-1:i1]
  rhoR[i0:i1+1] = rho[i0:i1+1, RK] - 0.5 * dx * s[i0:i1+1]
  
  # velocity states
  sa[0] = 0
  sa[i0-1:i1+2] = (v[i0-1:i1+2] - v[i0-2:i1+1]) / dx
  sb[i0-1:i1+1] = sa[i0:i1+2]
  
  s[:] = 0 # by default
  
  entries = (sa < 0) & (sb < 0)
  s[entries] = np.maximum(sa[entries], sb[entries])
  
  entries = (sa > 0) & (sb > 0)
  s[entries] = np.minimum(sa[entries], sb[entries])
  
  vL[i0:i1+1] = v[i0-1:i1] + 0.5 * dx * s[i0-1:i1]
  vR[i0:i1+1] = v[i0:i1+1] - 0.5 * dx * s[i0:i1+1]
  
  # pressure states
  sa[0] = 0
  sa[i0-1:i1+2] = (p[i0-1:i1+2] - p[i0-2:i1+1]) / dx
  sb[i0-1:i1+1] = sa[i0:i1+2]
  
  s[:] = 0 # by default
  
  entries = (sa < 0) & (sb < 0)
  s[entries] = np.maximum(sa[entries], sb[entries])
  
  entries = (sa > 0) & (sb > 0)
  s[entries] = np.minimum(sa[entries], sb[entries])
  
  pL[i0:i1+1] = p[i0-1:i1] + 0.5 * dx * s[i0-1:i1]
  pR[i0:i1+1] = p[i0:i1+1] - 0.5 * dx * s[i0:i1+1]
  
  # derive the other states using EOS (this includes a local prim2cons version)
  rhovL[i0:i1+1] = rhoL[i0:i1+1] * vL[i0:i1+1]
  rhovR[i0:i1+1] = rhoR[i0:i1+1] * vR[i0:i1+1]
  EL[i0:i1+1] = 0.5 * vL[i0:i1+1] * rhovL[i0:i1+1] + pL[i0:i1+1] / (gamma - 1.)
  ER[i0:i1+1] = 0.5 * vR[i0:i1+1] * rhovR[i0:i1+1] + pR[i0:i1+1] / (gamma - 1.)

def set_flux(RK = 0):
  # set states immediately to left and right of cell boundary.
  set_LR_states(RK)
  
  # fluxes to left and right
  FrhoL[i0:i1+1] = rhovL[i0:i1+1]
  FrhovL[i0:i1+1] = rhovL[i0:i1+1] * vL[i0:i1+1] + pL[i0:i1+1]
  FEL[i0:i1+1] = (EL[i0:i1+1] + pL[i0:i1+1]) * vL[i0:i1+1]

  FrhoR[i0:i1+1] = rhovR[i0:i1+1]
  FrhovR[i0:i1+1] = rhovR[i0:i1+1] * vR[i0:i1+1] + pR[i0:i1+1]
  FER[i0:i1+1] = (ER[i0:i1+1] + pR[i0:i1+1]) * vR[i0:i1+1]
  
  # sound speeds to left and right
  csL[i0:i1+1] = np.sqrt(gamma * pL[i0:i1+1] / rhoL[i0:i1+1])
  csR[i0:i1+1] = np.sqrt(gamma * pR[i0:i1+1] / rhoR[i0:i1+1])
  
  # HLL-specific quantity pstar
  pstar[i0:i1+1] = (0.5 * (pL[i0:i1+1] + pR[i0:i1+1]) - 0.5 * 
    (vR[i0:i1+1] - vL[i0:i1+1]) * 0.5 * (rhoL[i0:i1+1] + rhoR[i0:i1+1]) * 0.5 *
    (csL[i0:i1+1] + csR[i0:i1+1]))
  # there are some unused entries in the arrays above, corresponding to ghost
  # cell positions that do not need flux computation. Rather than defining
  # shorter arrays without unused entries, I opted for this approach which
  # ensures that the relation between an entry index and a grid position is
  # always the same, regardless of which array we consider. Again, the aim
  # is clear code, while the cost in extra memory is genuinely negligible.
  # We do pay a price when accessing entries using a conditional statement,
  # which would check for this condition across all entries normally, so we need
  # to add a mask for flux_entries that would otherwise not be needed
  pstar[(pstar < 0) & flux_entries] = 0.0
  
  # set the wave speeds SL
  entries = (pstar <= pL) & flux_entries
  SL[entries] = vL[entries] - csL[entries]
  entries = (pstar > pL) & flux_entries
  SL[entries] = vL[entries] - csL[entries] * np.sqrt(1 + (gamma + 1) / 
    (2. * gamma) * (pstar[entries]/pL[entries] - 1))

  # set the wave speeds SR
  entries = (pstar <= pR) & flux_entries
  SR[entries] = vR[entries] + csR[entries]
  entries = (pstar > pR) & flux_entries
  SR[entries] = vR[entries] + csR[entries] * np.sqrt(1 + (gamma + 1) / 
    (2. * gamma) * (pstar[entries]/pR[entries] - 1))
  
  # set flux across boundary option one, both waves move to the right
  entries = flux_entries & (SL > 0) & (SR > 0)
  Frho[entries] = FrhoL[entries]
  Frhov[entries] = FrhovL[entries]
  FE[entries] = FEL[entries]
  
  # set flux across boundary option two, both waves move to the left 
  entries = flux_entries & (SL < 0) & (SR < 0)
  Frho[entries] = FrhoR[entries]
  Frhov[entries] = FrhovR[entries]
  FE[entries] = FER[entries]
  
  # set flux across boundary option three, draw upon starred HLL state
  entries = ~((SL > 0) & (SR > 0)) & ~((SL < 0) & (SR < 0)) & flux_entries
  Frho[entries] = (SR[entries] * FrhoL[entries] - SL[entries] * FrhoR[entries]
    + SR[entries] * SL[entries] * (rhoR[entries] - rhoL[entries])) / (
    SR[entries] - SL[entries])
  Frhov[entries] = (SR[entries] * FrhovL[entries] - SL[entries] *FrhovR[entries]
    + SR[entries] * SL[entries] * (rhovR[entries] - rhovL[entries])) / (
    SR[entries] - SL[entries])
  FE[entries] = (SR[entries] * FEL[entries] - SL[entries] * FER[entries]
    + SR[entries] * SL[entries] * (ER[entries] - EL[entries])) / (
    SR[entries] - SL[entries])

def set_dt():
  # set the allowed timestep according to CFL condition
  dt_local[i0:i1] = dx / (np.absolute(v[i0:i1]) + 
    np.sqrt(gamma * p[i0:i1] / rho[i0:i1, 0]))
  global dt
  global finished
  dt = 0.3 * np.min(dt_local[i0:i1]) # a margin of 0.3
  if (t + dt > t1):
    dt = (t1 - t)
    finished = True

def update_grid(RKstep = -1):
  
  if RKstep == -1: # revert to Forward Euler, included for comparison
    rho[i0:i1, 0] = rho[i0:i1, 0] - dt / dx * (Frho[i0+1:i1+1] - Frho[i0:i1])
    rhov[i0:i1, 0] = rhov[i0:i1, 0] - dt/dx * (Frhov[i0+1:i1+1] - Frhov[i0:i1])
    E[i0:i1, 0] = E[i0:i1, 0] - dt / dx * (FE[i0+1:i1+1] - FE[i0:i1])
  
  # Third-order RK scheme for function dy/dt = f(t, y):
  # 
  # tableau: 0   | 0    0    0        c_0 | a_00 a_01 a_02
  #          1   | 1    0    0        c_1 | a_10 a_11 a_12
  #          1/2 | 1/4  1/4  0        c_2 | a_20 a_21 a_22
  #          ----+---------------    -----+----------------
  #              | 1/6  1/6  2/3          | b_0  b_1  b_2
  #
  # y_{n+1} = y_n + h SUM_{i=0}^1 b_i k_i
  #
  # k_i = f( t_n + c_i h, y_n + h SUM_{j=0}^{i-1} a_ij k_j )
  # 
  # in our case, f not a direct function of t, so c_i will not be needed
  # I start counting all indices at zero, not one, to connect to programming
  # 
  # k_0 = f( y_n )
  # k_1 = f( y_n + h a_10 k_0 ) = f( y_n + h k_0 )
  # k_2 = f( y_n + h a_20 k_0 + h a_21 k_1 ) = f( y_n + (h/4) k_0 + (h/4) k_1 )
  #
  # Y_{n+1} = y_n + h b_0 k_0 + h b_1 k_1 + h b_2 k_2 
  #         = y_n + (h/6) k_0 + (h/6) k_1 + (2h/3) k_2
  #
  # In terms of intermediate states Q: 
  # Q_0 = y_n
  # k_0 = f( Q_0 )                             RK = 0 in set_flux routine
  # Q_1 = Q_0 + h f( Q_0 )                     RKstep == 0 below
  # k_1 = f( Q_1 )                             RK = 1 in set flux routine
  # Q_2 = Q_0 + (h/4) k_0 + (h/4) k_1          
  #     = (3/4) Q_0 + (1/4) Q_1 + (h/4) k_1    RKstep == 1 below
  # k_2 = f( Q_2 )                             RK = 2 in set flux routine
  # 
  # y_{n+1} = Q_0 + (h/6) k_0 + (h/6) k_1 + (2h/3) k_2  
  #         = (1/3) Q_0 + (2/3) Q_2 + (2h/3) f( Q_2 )      RKstep == 2 below
  #
  # We've used k_0 = (Q_1 - Q_0) / h and k_1 = (4 Q_2 - 3 Q_0 - Q_1) / h
  # to eliminate these terms from the equations for Q_2 and y_{n+1}, since we
  # are only storing the most recent set of fluxes at a given time
  
  if RKstep == 0: # first out of three-step third-order RK scheme
    rho[i0:i1, 1] = rho[i0:i1, 0] - dt / dx * (Frho[i0+1:i1+1] - Frho[i0:i1])
    rhov[i0:i1, 1] = rhov[i0:i1, 0] - dt/dx * (Frhov[i0+1:i1+1] - Frhov[i0:i1])
    E[i0:i1, 1] = E[i0:i1, 0] - dt / dx * (FE[i0+1:i1+1] - FE[i0:i1])
    
  if RKstep == 1: # second out of three-step third order RK scheme
    rho[i0:i1, 2] = (0.75 * rho[i0:i1, 0] + 0.25 * rho[i0:i1, 1]
        - 0.25 * dt / dx * (Frho[i0+1:i1+1] - Frho[i0:i1]))
    rhov[i0:i1, 2] = (0.75 * rhov[i0:i1, 0] + 0.25 * rhov[i0:i1, 1]
        - 0.25 * dt / dx * (Frhov[i0+1:i1+1] - Frhov[i0:i1]))
    E[i0:i1, 2] = (0.75 * E[i0:i1, 0] + 0.25 * E[i0:i1, 1] 
        - 0.25 * dt / dx * (FE[i0+1:i1+1] - FE[i0:i1]))

  if RKstep == 2: # third out of three-step third order RK scheme
    rho[i0:i1, 0] = (rho[i0:i1, 0]/3. + 2 * rho[i0:i1, 2]/3.
        - 2 * dt / dx * (Frho[i0+1:i1+1] - Frho[i0:i1])/3.)
    rhov[i0:i1, 0] = (rhov[i0:i1, 0]/3. + 2 * rhov[i0:i1, 2]/3.
        - 2 * dt / dx * (Frhov[i0+1:i1+1] - Frhov[i0:i1])/3.)
    E[i0:i1, 0] = (E[i0:i1, 0]/3. + 2 * E[i0:i1, 2]/3. 
        - 2 * dt / dx * (FE[i0+1:i1+1] - FE[i0:i1])/3.)
    
#-------------------------------------------------------------------------------
  
# INITIALIZATION: Set up a shock tube, using the primitive variables rho, v, p

xbound = 0.3
rho[x < xbound, 0] = 1.
v[x < xbound] = 0.75
p[x < xbound] = 1.
rho[x >= xbound] = 0.125 
v[x >= xbound] = 0.
p[x >= xbound] = 0.1


"""
xbound = 0.8
rho[x < xbound, 0] = 1.
v[x < xbound] = -19.59745
p[x < xbound] = 1000.
rho[x >= xbound] = 1. 
v[x >= xbound] = -19.59745
p[x >= xbound] = 0.01
t1 = 0.012 # overrule the value provided at the start of the source code
"""

prim2cons(0)

#-------------------------------------------------------------------------------
#itmax = 1

# run solver
while not finished:

  #print("iteration %d, t = %e:" % (iterations, t))

  # Forward Euler scheme, not in use by default
  #set_ghosts(0)
  #cons2prim(0)
  #set_dt()
  #set_flux(0)
  #update_grid(-1)
  
  # first round of third-order RK scheme
  set_ghosts(0)
  cons2prim(0)
  set_dt()
  set_flux(0)
  update_grid(0)
  
  # second round of third-order RK scheme
  set_ghosts(1)
  cons2prim(1)
  set_flux(1)
  update_grid(1)
  
  # third round of third-order RK scheme
  set_ghosts(2)
  cons2prim(2)
  set_flux(2)
  update_grid(2)
  
  t = t + dt
  iterations = iterations + 1
  
  # cap the total iterations
  if iterations == itmax:
    print("Maximum number of iterations (%d) reached" % iterations)
    finished = True

# make sure the primitive variables are also up to date  
cons2prim(0)

################################################################################
# Dump the output on the screen

for i in range(RES):
  print("%d, %e, %e, %e, %e, %e, %e" % 
    (i, x[no_ghosts+i] + 0.5*dx, rho[no_ghosts+i, 0], rhov[no_ghosts+i, 0], 
    E[no_ghosts+i, 0], v[no_ghosts+i], p[no_ghosts+i]))

################################################################################
# everything plotting related

if plot_output == True:
  import matplotlib.pyplot as plt
  import matplotlib.font_manager as fnt

  plt.rcParams['font.size'] = 15
  plt.rcParams['font.family'] = 'serif'
  fontprop = fnt.FontProperties()
  fontprop.set_size(13)

  plt.plot(x[grid_entries] + 0.5*dx, rho[grid_entries], color= 'blue', marker = '.')
  #plt.plot(x[grid_entries] + 0.5*dx, rhov[grid_entries], color= 'red', marker = '.')
  #plt.plot(x[grid_entries] + 0.5*dx, E[grid_entries], color= 'green', marker = '.')
  #plt.plot(x[grid_entries] + 0.5*dx, p[grid_entries], color= 'brown', marker = '.')
  #plt.plot(x[grid_entries] + 0.5*dx, v[grid_entries], color= 'black', marker = '.')

  plt.draw()
  plt.show()


