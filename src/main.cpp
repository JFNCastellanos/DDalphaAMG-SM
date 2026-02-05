#include <time.h> 
#include <ctime>
#include <fstream>
#include "conjugate_gradient.h"
#include "sap.h"
#include "tests.h"
#include "params.h"


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi::size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi::rank);
        
    //srand(mpi::rank*time(0));
    
    
    
    double m0; //bare mass
    
    CG::max_iter = 10000; //Maximum number of iterations for the conjugate gradient method
    CG::tol = 1e-10; //Tolerance for convergence


    

    //To call the sequential program one has to choose ranks_x = ranks_t = 1
    if (mpi::rank == 0){
         //---Input data---//
        std::cout << " -----------------------------" << std::endl;
        std::cout << "|  DDα-AMG Schwinger Model     |" << std::endl;
        std::cout << " -----------------------------" << std::endl;
        std::cout << "Nx " << LV::Nx << " Nt " << LV::Nt << std::endl;
        std::cout << "ranks_x: ";
        std::cin >> mpi::ranks_x;
        std::cout << "ranks_t: ";
        std::cin >> mpi::ranks_t;
        std::cout << "Levels: ";
        std::cin >> LevelV::levels;
        std::cout << "m0: ";
        std::cin >> m0;
       
    }

    MPI_Bcast(&mpi::ranks_x, 1, MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&mpi::ranks_t, 1, MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&LevelV::levels, 1, MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&m0, 1, MPI_DOUBLE,  0, MPI_COMM_WORLD);
    

    //The order in which these functions are called is very important
    initializeMPI(); //2D rank topology
    LevelV::maxLevel = LevelV::levels-1;
    allocate_lattice_arrays(); 
    readParameters("../inputs");
    boundaries();


    SAP_fine_level sap(mpi::width_x,  mpi::width_t, SAPV::sap_block_x, SAPV::sap_block_t, 2, 1);

    srand((mpi::rank2d+1));
       

    
    spinor U(mpi::maxSizeH);
    spinor phi(mpi::maxSizeH);

    for(int x = 1; x<=mpi::width_x; x++){
        for(int t = 1; t<=mpi::width_t; t++){
            int n = x*(mpi::width_t+2)+t;
            U.val[2*n]      = RandomU1();
            U.val[2*n+1]    = RandomU1();
            phi.val[2*n]      = RandomU1();
            phi.val[2*n+1]    = RandomU1();
        }
    }

    int l = 0;
    //AssembleP_Pdagg(l,U);
    //Check_PPdagg(l,U);

    



    //Free coordinate arrays
    free_lattice_arrays();
    MPI_Finalize();

	return 0;
}

