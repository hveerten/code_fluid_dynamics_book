################################################################################
#
# plot_vectorfield_2D.py
#
# Hendrik van Eerten, University of Bath.
# 2020-09-23 
#
################################################################################

# Import relevant modules
import os
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams["savefig.directory"] = os.path.dirname(__file__)
matplotlib.rcParams.update({'font.size': 18})
import matplotlib.pyplot as plt
import numpy as np

#-------------------------------------------------------------------------------

# Set the resolution for the plot
domain_size = 1.
res_x = domain_size / 8.
res_y = domain_size / 8.

epsilon = 0.001 * domain_size

#-------------------------------------------------------------------------------

# Set up the coordinates, both in Cartesian and polar coordinate systems
x_line = np.arange(-domain_size - epsilon, domain_size + epsilon, res_x)
y_line = np.arange(-domain_size - epsilon, domain_size + epsilon, res_y)
x, y = np.meshgrid(x_line, y_line)

r = np.sqrt(x*x + y*y)
phi = np.arctan(y / x) # arctan uses -0.5pi .. 0.5pi which is not what we need

# correct for the other quadrants, generate a phi in 0.. 2pi
subset = np.logical_and(x < 0, y >= 0)
phi[subset] = np.arctan(-x[subset] / y[subset]) + 0.5 * np.pi
subset = np.logical_and(x < 0, y < 0)
phi[subset] = np.arctan(-y[subset] / -x[subset]) + 1.0 * np.pi
subset = np.logical_and(x >= 0, y < 0)
phi[subset] = np.arctan(x[subset] / -y[subset]) + 1.5 * np.pi

# set up the unit vectors
e_x_x = 1.
e_x_y = 0.
e_y_x = 0.
e_y_y = 1.

e_r_x = np.cos(phi)
e_r_y = np.sin(phi)

e_phi_x = -np.sin(phi)
e_phi_y = np.cos(phi)

#-------------------------------------------------------------------------------
# Set up a vector field of interest to be plotted

v_x = e_r_x / r
v_y = e_r_y / r

#-------------------------------------------------------------------------------

# Set up the actual plot and draw it

fig, ax = plt.subplots()

subset = r > 0.1 # pick out a subset of the grid for inclusion in the plot

# code for a 2D plot.
plt.axvline(0, color='grey', zorder=1)
plt.axhline(0, color='grey', zorder=1)
q = ax.quiver(x[subset], y[subset], v_x[subset], v_y[subset], zorder=2)

plt.xlim(-1, 1)
plt.ylim(-1, 1)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

plt.show()
