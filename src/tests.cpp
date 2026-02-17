#include "tests.h"

void test_gather_Datatypes_level_class(Level& lev,const int dof){
    if (mpi::size != 16){
         printf("This test is meant to be run with 16 processes.\n");
        MPI_Abort(mpi::cart_comm, EXIT_FAILURE);
    }
    
    spinor input((lev.Nt+2)*(lev.Nx+2)*dof);
    int n;
    for(int x = 1; x<=lev.Nx; x++){
        for(int t = 1; t<=lev.Nt; t++){
            n = x*(lev.Nt+2)+t;
            for(int mu=0; mu<dof; mu++){
                input.val[dof*n+mu] = dof*n+mu + mpi::rank2d;
            }        
        }
    }


    for(int i = 0; i < mpi::size; i++) {
        MPI_Barrier(mpi::cart_comm);
        if (i == mpi::rank2d) {
            std::cout << "rank " << mpi::rank2d << std::endl;
            for(int x = 1; x<=lev.Nx; x++){
                for(int t = 1; t<=lev.Nt; t++){
                    int n = x*(lev.Nt+2) + t;
                    std::cout << "[";
                    for(int mu = 0; mu<dof; mu++){
                         std::cout << input.val[dof*n+mu] << ", ";
                    }   
                    std::cout << "], ";
                }
                std::cout << std::endl;
            } 
        }
    }

  
    spinor buffer((lev.Nx_coarse_rank+2)*(lev.Nt_coarse_rank+2)*dof);

    lev.gather_to_coarse_rank(input, buffer,dof);

   
    int local_rank;
    int commID = mpi::rank_dictionary[mpi::rank2d];  
    MPI_Comm_rank(mpi::coarse_comm[commID], &local_rank);
    for(int i = 0; i <mpi::size; i++) {
        MPI_Barrier(mpi::cart_comm);
        if (i == mpi::rank2d && local_rank == 0) {
            std::cout << "\nGather in rank " << mpi::rank2d << std::endl;
            for(int x = 0; x<lev.Nx_coarse_rank+2; x++){
                for(int t = 0; t<lev.Nt_coarse_rank+2; t++){
                    int n = x*(lev.Nt_coarse_rank+2) + t;
                    std::cout << "[";
                    for(int mu = 0; mu<dof; mu++){
                         std::cout << buffer.val[dof*n+mu] << ", ";
                    }   
                    std::cout << "], ";
                }
                std::cout << std::endl;
            } 
        }
    }
}

void gather_tests(){
    spinor U(mpi::maxSizeH);
    int l = 0;
    Level lev(l,U);
    int DOF = lev.DOF;
    if (mpi::rank2d == 0)
        std::cout << "Test gather for spinor at level " << l << " with DOF=" << DOF << std::endl;
    test_gather_Datatypes_level_class(lev,DOF);
     if (mpi::rank2d == 0)
        std::cout << "Test gather for G1 at level " << l << " with DOF=" << DOF << std::endl;
    test_gather_Datatypes_level_class(lev,DOF*DOF);
     if (mpi::rank2d == 0)
        std::cout << "Test gather for G2G3 at level " << l << " with DOF=" << DOF << std::endl;
    test_gather_Datatypes_level_class(lev,DOF*DOF*2);
}