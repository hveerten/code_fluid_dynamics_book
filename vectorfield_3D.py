################################################################################
#
# plot_vectorfield_3D.py
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

#-------------------------------------------------------------------------------

# Set the resolution for the plot
domain_size = 1.
res_x = domain_size / 4.
res_y = domain_size / 4.
res_z = domain_size / 4.

epsilon = 0.001 * domain_size # make the plot boundaries inclusive

#-------------------------------------------------------------------------------

# Set up the coordinates, in Cartesian, spherical and cylindrical systems. These
# can be used to help define the vector field

x_line = np.arange(-domain_size - epsilon, domain_size + epsilon, res_x)
y_line = np.arange(-domain_size - epsilon, domain_size + epsilon, res_y)
z_line = np.arange(-domain_size - epsilon, domain_size + epsilon, res_z)
x, y, z = np.meshgrid(x_line, y_line, z_line, indexing='ij')

r = np.sqrt(x*x + y*y + z*z)
theta = np.arccos(z / r)
phi = np.arctan(y / x) # arctan uses -0.5pi .. 0.5pi which is not what we need
h = np.sqrt(x*x + y*y) # distance from z-axis (Cylindrical coordinates)

# correct for the other quadrants, generate a phi in 0.. 2pi
subset = np.logical_and(x < 0, y >= 0)
phi[subset] = np.arctan(-x[subset] / y[subset]) + 0.5 * np.pi
subset = np.logical_and(x < 0, y < 0)
phi[subset] = np.arctan(-y[subset] / -x[subset]) + 1.0 * np.pi
subset = np.logical_and(x >= 0, y < 0)
phi[subset] = np.arctan(x[subset] / -y[subset]) + 1.5 * np.pi

#-------------------------------------------------------------------------------

# Set up the unit vectors in Cartesian, spherical and cylindrical systems. These
# too can be used to help define the vector field

e_x_x = 1.
e_x_y = 0.
e_x_z = 0.

e_y_x = 0.
e_y_y = 1.
e_y_z = 0.

e_z_x = 0.
e_z_y = 0.
e_z_z = 1.

e_r_x = np.cos(phi) * np.sin(theta)
e_r_y = np.sin(phi) * np.sin(theta)
e_r_z = np.cos(theta)

e_theta_x = np.cos(theta) * np.cos(phi)
e_theta_y = np.cos(theta) * np.sin(phi)
e_theta_z = -np.sin(theta)

e_phi_x = - np.sin(theta) * np.sin(phi)
e_phi_y = np.sin(theta) * np.cos(phi)
e_phi_z = 0 * theta # theta only used to set the dimensions of the array

e_h_x = np.cos(phi)
e_h_y = np.sin(phi)
e_h_z = 0.

#-------------------------------------------------------------------------------
# Set up a vector field of interest to be plotted

v_x = 0.1 * e_r_x / h
v_y = 0.1 * e_r_y / h
v_z = 0.1 * e_r_z / h

#-------------------------------------------------------------------------------

# Set up the actual plot and draw it

fig = plt.figure()
ax = fig.gca(projection='3d')

subset = h > 0.1 # pick out a subset of the grid for inclusion in the plot

# code for a 3D plot
q = ax.quiver(x[subset], y[subset], z[subset], v_x[subset], v_y[subset], 
  v_z[subset], color = 'black')

plt.show()
