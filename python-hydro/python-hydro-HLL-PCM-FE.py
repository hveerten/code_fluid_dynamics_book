# python-hydro
#
# An implementation of a finite volume 1D hydro solver in python, done as simple
# as possible to demonstrate the CFD concepts rather than python.
#
# This version uses the HLL method, the Piecewise Constant Method and a 
# Forward Euler time step.

import numpy as np

# resolution settings
RES = 200 # set the numerical resolution, excluding ghost cells
no_ghosts = 1 # number of ghost cells
itmax = 100000 # maximum number of iterations, use negative number to ignore

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
rho = np.empty(RES + 2 * no_ghosts) # conserved variable density
rhov = np.empty(RES + 2 * no_ghosts) # conserved variable momentum density
E = np.empty(RES + 2 * no_ghosts) # conserved variable energy density

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
# defined just to the right of the left cell boundary of cell i. In our simple
# case of piecewise flat cell content, we don't really need a separate L and R
# quantity, but this sets the stage for higher order spatial reconstruction
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

#-------------------------------------------------------------------------------
 
def prim2cons():
  # compute conserved quantities rho, S, E based on primitive rho, p, v
  # (rho is both, so does not need separate computing). 
  # Only acts on non-ghost cells.
  rhov[i0:i1] = rho[i0:i1] * v[i0:i1]
  E[i0:i1] = 0.5 * v[i0:i1] * rhov[i0:i1] + p[i0:i1] / (gamma - 1.)
  
def cons2prim():
  # compute primitive quantities rho, p, v based on conserved rho, S, E
  # (rho is both, so does not need separate computing).
  # Acts on ghost cells as well
  v[ig0:ig1] = rhov[ig0:ig1] / rho[ig0:ig1]
  p[ig0:ig1] = (gamma - 1.) * (E[ig0:ig1] - 0.5 * rhov[ig0:ig1] * v[ig0:ig1]) 

def set_ghosts():
  # set the conserved quantity values of the ghost cells
  for i in range(no_ghosts):
 
    # inflow/outflow BC left side
    rho[i] = rho[no_ghosts]
    rhov[i] = rhov[no_ghosts]
    E[i] = E[no_ghosts]
    
    # inflow/outflow BC right side
    rho[i1 + i] = rho[i1 - 1]
    rhov[i1 + i] = rhov[i1 - 1]
    E[i1 + i] = E[i1 - 1]

    # reflecting BC
    #rho[i] = rho[no_ghosts]
    #rhov[i] = -rhov[no_ghosts]
    #E[i] = E[no_ghosts]
    #rho[i1 + i] = rho[i1 - 1]
    #rhov[i1 + i] = -rhov[i1 - 1]
    #E[i1 + i] = E[i1 - 1]


def set_flux():
  # set states immediately to left and right of cell boundary.
  pL[i0:i1+1] = p[i0-1:i1]
  vL[i0:i1+1] = v[i0-1:i1]

  rhoL[i0:i1+1] = rho[i0-1:i1]
  rhovL[i0:i1+1] = rhov[i0-1:i1]
  EL[i0:i1+1] = E[i0-1:i1]

  pR[i0:i1+1] = p[i0:i1+1]
  vR[i0:i1+1] = v[i0:i1+1]

  rhoR[i0:i1+1] = rho[i0:i1+1]
  rhovR[i0:i1+1] = rhov[i0:i1+1]
  ER[i0:i1+1] = E[i0:i1+1]

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
    np.sqrt(gamma * p[i0:i1] / rho[i0:i1]))
  global dt
  global finished
  dt = 0.3 * np.min(dt_local[i0:i1]) # a margin of 0.3
  if (t + dt > t1):
    dt = (t1 - t)
    finished = True

def update_grid():
  rho[i0:i1] = rho[i0:i1] - dt / dx * (Frho[i0+1:i1+1] - Frho[i0:i1])
  rhov[i0:i1] = rhov[i0:i1] - dt / dx * (Frhov[i0+1:i1+1] - Frhov[i0:i1])
  E[i0:i1] = E[i0:i1] - dt / dx * (FE[i0+1:i1+1] - FE[i0:i1])
  
#-------------------------------------------------------------------------------
  
# INITIALIZATION: Set up a shock tube, using the primitive variables rho, v, p

xbound = 0.3
rho[x < xbound] = 1.
v[x < xbound] = 0.75
p[x < xbound] = 1.
rho[x >= xbound] = 0.125 
v[x >= xbound] = 0.
p[x >= xbound] = 0.1


"""
xbound = 0.8
rho[x < xbound] = 1.
v[x < xbound] = -19.59745
p[x < xbound] = 1000.
rho[x >= xbound] = 1. 
v[x >= xbound] = -19.59745
p[x >= xbound] = 0.01
t1 = 0.012 # overrule the value provided at the start of the source code
"""

prim2cons()

#-------------------------------------------------------------------------------

# run solver
while not finished:

  #print("iteration %d, t = %e:" % (iterations, t))
  set_ghosts()
  cons2prim()
  set_dt()
  set_flux()
  
  update_grid()
  t = t + dt
  iterations = iterations + 1
  
  # cap the total iterations
  if iterations == itmax:
    print("Maximum number of iterations (%d) reached" % iterations)
    finished = True
  

################################################################################
# everything plotting related
import matplotlib.pyplot as plt
import matplotlib.font_manager as fnt

plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = 'serif'
fontprop = fnt.FontProperties()
fontprop.set_size(13)

plt.plot(x[grid_entries] + 0.5*dx, rho[grid_entries], color= 'blue', marker = '.')
#plt.plot(x[grid_entries] + 0.5*dx, rhov[grid_entries], color= 'red')
#plt.plot(x[grid_entries] + 0.5*dx, E[grid_entries], color= 'green')
#plt.plot(x[grid_entries] + 0.5*dx, p[grid_entries], color= 'brown')
#plt.plot(x[grid_entries] + 0.5*dx, v[grid_entries], color= 'black')


plt.draw()
plt.show()


