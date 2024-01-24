// IO.h

#ifndef IO_
#define IO_

#include <stdio.h>

#include "grid.h"
#include "settings.h"

void print_state(grid *g);
// prints the state to screen
// Arguments:
//   g  struct with grid info

void print_state_ghosts(grid *g);
// prints the state to screen, includes ghost cells
// Arguments:
//   g  struct with grid info

void fprint_state(const char* filename, grid *g);
// prints the state to a file
// Arguments:
//   filename   filename to use
//   g          struct with grid info

#endif
