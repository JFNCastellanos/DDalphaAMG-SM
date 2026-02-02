cmake_minimum_required(VERSION 3.26.5)
set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CXX_STANDARD_REQUIRED True)
project(SAP VERSION 1.0)


set(SOURCES
    src/variables.cpp           #Relevant parameters for the lattice
    src/utils.cpp               #Utility functions
    src/halo_exchange.cpp       #Implementation of the halo exchange for a field
    src/dirac_operator.cpp      #Implementation of D, D^+ and 2* Re ( left^+ d D / d omega(z) right )
    src/conjugate_gradient.cpp  #Conjugate gradient and Bi-cgstab
    src/fgmres.cpp
    src/sap.cpp
    src/main.cpp                #Main program
)

set(NS "1024")   #Lattice dimension x
set(NT "1024")   #Lattice dimension t

configure_file(${CMAKE_SOURCE_DIR}/include/config.h.in ${CMAKE_SOURCE_DIR}/include/config.h)


add_executable("SM_${NS}x${NT}" ${SOURCES})


target_include_directories("SM_${NS}x${NT}"
        PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
)
target_compile_options("SM_${NS}x${NT}" PRIVATE
			-O3
            #-g #For profiling
)


find_package(MPI REQUIRED)
target_link_libraries("SM_${NS}x${NT}" PUBLIC MPI::MPI_CXX)


#PROJECT_SOURCE_DIR: it indicates where src is located
#cmake -G "MinGW Makefiles" -DCMAKE_CXX_COMPILER="C:/msys64/ucrt64/bin/g++.exe" -DCMAKE_C_COMPILER="C:/msys64/ucrt64/bin/gcc.exe" ../
#dir/s/b *.txt > confs.txt