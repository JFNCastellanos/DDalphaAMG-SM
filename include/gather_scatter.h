#ifndef GATHER_SCATTER_H
#define GATHER_SCATTER_H

#include "variables.h"

/*
Implementation of functions that gather (scatter) data from local (coarse) domains to a 
rank of the coarse (local) communicator. 

initializeMPI() should be called before using the functions
*/


//Gather from information of the ranks on the fine grid to a particular rank in the coarse_communicator
inline void gather_to_coarse_rank(const spinor& local_spinor, spinor& coarse_spinor){
	int commID = mpi::rank_dictionary[mpi::rank2d];  
    static int input_ini_local = 2 * (mpi::width_t + 2 + 1); // start of [1,1] in local input (complex elements)
    static int root_rank = 0;  //Root rank inside each communicator agglomerating ranks

    // Use Gatherv: send 1 instance of the local strided type, receive into the resized global type
    MPI_Gatherv(&local_spinor.val[input_ini_local],
                1,
                local_domain,
                &coarse_spinor.val[0],
                mpi::counts_coarse,
                mpi::displs_coarse,
                coarse_domain_resized,
                root_rank,
                mpi::coarse_comm[commID]);
}

//Scatter information from the root rank of the coarse communicator to ranks on the fine grid
inline void scatter_to_local_rank_from_coarse_rank(const spinor& coarse_spinor, spinor& local_spinor){
	int commID = mpi::rank_dictionary[mpi::rank2d];  
    static int input_ini_local = 2 * (mpi::width_t + 2 + 1); // start of [1,1] in local input (complex elements)
    static int root_rank = 0;  //Root rank inside each communicator agglomerating ranks

    MPI_Scatterv(&coarse_spinor.val[0],
        mpi::counts_coarse,
        mpi::displs_coarse,
        coarse_domain_resized,
        &local_spinor.val[input_ini_local],
        1,
        local_domain_resized,
        root_rank,
        mpi::coarse_comm[commID]);
}

#endif