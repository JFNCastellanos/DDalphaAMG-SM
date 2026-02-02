#ifndef HALO_EXCHANGE_H
#define HALO_EXCHANGE_H
#include "mpi_setup.h"

//Phi has dimension [2*(width_x+2)*(width_t+2)]
void exchange_halo(c_double* phi);


#endif