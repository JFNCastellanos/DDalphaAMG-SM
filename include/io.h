#ifndef IO_H
#define IO_H

#include "mpi_setup.h"
#include <string>
#include <fstream>

//Function for reading configurations or rhs
void read_binary(const std::string& name,const spinor& U);

#endif