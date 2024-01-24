################################################################################
#
# makefile
#
################################################################################

# compiler
CXX = gcc-10
#CXX = pgcc

# Linker flags:
LDFLAGS = -lm -lgomp
#LDFLAGS = -lm -acc
#-foffload=nvptx-none

# Compiler flags
CXXFLAGS = -Wall -O3 -march=native -mtune=native -fopenmp
#CXXFLAGS = -O3 -fast -acc=gpu -Minline
#XXFLAGS = -g -O0 -acc
#CXXFLAGS = -Wall -O0 -g

#CPPFLAGS = -I/usr/include/mpi -I/usr/include/hdf5/openmpi -Wall -O0 -g -fopenmp
#CPPFLAGS = -I/usr/include/mpi -I/usr/include/hdf5/openmpi -Wall -O3 -march=native -mtune=native
  # -Wall all warnings on
  # -O3 Optimization level 3 ( -O0 needed for debugging purposes )
  # ad -g to compile for debugging purposes. Debug code using valgrind
  # -fopenmp enable openMP features
  # -march=native and -mtune=native perform processor-specific optimization

################################################################################

SOURCE  = arraytools.c consprim.c initial_conditions.c IO.c boundary.c \
  reconstruct.c solver.c source.c toy-hydro.c

OBJECTS = ${SOURCE:.c=.o}                           

################################################################################

TOY-HYDRO_O = arraytools.o consprim.o initial_conditions.o IO.o boundary.o \
  reconstruct.o solver.o source.o toy-hydro.o
  
################################################################################

.c.o: ${SOURCE}
	${CXX} -c ${CXXFLAGS} $<                             

toy-hydro: ${TOY-HYDRO_O}
	${CXX} -o ../bin/$@ ${TOY-HYDRO_O} ${LDFLAGS}

all: toy-hydro

clean:                                                
	rm -f *.o
