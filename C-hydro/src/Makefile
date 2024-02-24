################################################################################
#
# makefile
#
################################################################################

# compiler
CXX = gcc-10 # a standard open-source C compiler 
#CXX = pgcc # The NVIDIA ANSI and K&R C compiler, helpful for GPU functionality

# Compiler flags
CXXFLAGS = -Wall -O3 -march=native -fopenmp -mtune=native # gcc-10 version
#CXXFLAGS = -Wall -O3 -acc -march=native # pgcc version
  # -Wall         all warnings on
  # -O3           Optimization level 3 ( -O0 needed for debugging purposes )
  # -g            Compile for debugging purposes. Debug code using e.g. valgrind
  # -fopenmp      enable openMP features
  # -march=native perform processor-specific optimization
  # -mtune=native perform processor-specific optimization (for gcc-10)
  # -acc          includes GPU parallelization through openacc (for pgcc)

# Linker flags:
LDFLAGS = -lm -lgomp # gcc-10 version
#LDFLAGS = -lm -acc # pgcc version
  # Common flags:
  # -lm    includes the math library
  # -lgomp includes openMP library to access routines 
  # -acc   includes openacc library to access routines (for pgcc)

################################################################################

SOURCE  = arraytools.c consprim.c initial_conditions.c IO.c boundary.c \
  reconstruct.c solver.c source.c C-hydro.c

OBJECTS = ${SOURCE:.c=.o}                           

################################################################################

C-HYDRO_O = arraytools.o consprim.o initial_conditions.o IO.o boundary.o \
  reconstruct.o solver.o source.o C-hydro.o
  
################################################################################

.c.o: ${SOURCE}
	${CXX} -c ${CXXFLAGS} $<                             

C-hydro: ${C-HYDRO_O}
	${CXX} -o ../bin/$@ ${C-HYDRO_O} ${LDFLAGS}

all: C-hydro

clean:                                                
	rm -f *.o
