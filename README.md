# Introduction to Fluid Dynamics in Physics and Astrophysics 
## H.J. van Eerten (2024, CRC Press, Taylor & Francis Group)

This repository includes code and scripts to accompany the textbook [Introduction to Fluid Dynamics in Physics and Astrophysics](https://www.taylorfrancis.com/books/mono/10.1201/9781003095088/introduction-fluid-dynamics-physics-astrophysics-hendrik-jan-van-eerten). Currently the following content is included:

### Errata

The file errata.pdf contains a list of typos in the book.

### Various scripts

A number of python scripts to experiment with when learning the concepts of the book can be found in the folder "various_scripts".

### A python version of a hydrodynamics solver

The folder python-hydro contains various iterations of a one-dimensional hydrodynamics solver using the finite volume method as described in chapter 13 of the book, programmed in python. It should run a shock-tube problem out of the box. The solvers are written to convey the concepts of chapter 13 as clearly as possible, rather than to optimize efficiency in e.g. memory management. The simplest implementation can be found in python-hydro-HLL-PCM-FE.py, the most sophisticated in python-hydro-HLLC-PLM-RK3.py. The filename components have the following meaning:

<ul>
  <li>HLL    HLL-solver</li>
  <li>HLLC   HLLC-solver</li>
  <li>PCM    Piecewise Constant Method</li>
  <li>PLM    Piecewise Linear Method</li>
  <li>FE     Forward Euler</li>
  <li>RK     Runge-Kutta</li>
</ul>

### A C hydrodynamics solver

The folder C-hydro contains a multi-dimensional hydrodynamics solver using the HLLC finite volume method as described in Chapter 13 of the book, written in C. Unlike the python versions, this one was written with performance in mind rather than ease of reading. It can run in serial (default setting), in parallel using openMP, or in parallel using openACC (that is, using GPU parallelization). When compiled it will run the same one-dimensional shock-tube problem as the python solver by default. The number of dimensions can be increased to two or three. In addition to the shock tube, three other problems are provided: a Kelvin-Helmholtz instability, a Rayleigh-Taylor instability and a demonstration using a source term, in the form of an externally imposed gravity force towards the centre of the grid.

#### setting up and compiling C-hydro

The source code can all be found in the /src subdirectory. First check the Makefile to see if you need to make any changes to the paths and compiler commands. Then compile as usual with "make all" or "make C-hydro". By default, this will compile a serial implementation of the standard one-dimensional shock-tube problem. An executable "C-hydro" will be written to the /bin directory.
In order to change any of the settings (computational method, resolution, which set of initial conditions to use, etc.), modify the content of "settings.h" before compiling. To set up your own initial conditions, the file "initial_conditions.c" can be modified or replaced altogether. Both settings.h and initial_conditions.c are intended to be self-explanatory through their annotation.
