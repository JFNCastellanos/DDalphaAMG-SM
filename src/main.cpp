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
        std::cout << "|   DDalpha-AMG Schwinger Model   |" << std::endl;
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
    mass::m0 = m0;

    //The order in which these functions are called is very important

    LevelV::maxLevel = LevelV::levels-1;
    allocate_lattice_arrays(); 
    initializeMPI(); //2D rank topology
    readParameters("../inputs");
    boundaries();

    srand((mpi::rank2d+1));
    
    spinor U(mpi::maxSizeH);


    int l = 0;
    //AssembleP_Pdagg(l,U);
    //Check_PPdagg(l,U);
    //test_Doperator_fine_level(U);
    //rank_agglomeration_test();

    
    //gather_vector_test();
    //scatter_vector_test();

    //coarse_gauge_links_test(U);
    //test_Dc(U);
    //check_boundaries(U);
    //test_P_vc_rank_coarsening(U);
    //test_Pdagg_rank_coarsening(U);
    //Check_PPdagg_coarsening(U);

    //test_PPdagg_any_case(U);

    //gather_tests();
    //scatter_tests();

    //test_Dc_with_rank_coarsening();
    //test_SAP_in_level_0();
    test_SAP_in_every_level();

    //Free coordinate arrays
    free_lattice_arrays();
    MPI_Finalize();

	return 0;
}