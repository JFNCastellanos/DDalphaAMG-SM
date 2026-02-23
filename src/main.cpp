#include <time.h> 
#include <ctime>
#include <fstream>
#include "conjugate_gradient.h"
#include "sap.h"
#include "tests.h"
#include "params.h"
#include "methods.h"


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
        std::cout << "| DDalpha-AMG Schwinger Model |" << std::endl;
        std::cout << " -----------------------------" << std::endl;
        std::cout << "Nx " << LV::Nx << " Nt " << LV::Nt << std::endl;
        std::cout << "ranks_x: ";
        std::cin >> mpi::ranks_x;
        std::cout << "ranks_t: ";
        std::cin >> mpi::ranks_t;
        std::cout << "Levels: ";
        std::cin >> LevelV::levels;
        //std::cout << "m0: ";
        //std::cin >> m0;
       
    }

    MPI_Bcast(&mpi::ranks_x, 1, MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&mpi::ranks_t, 1, MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&LevelV::levels, 1, MPI_INT,  0, MPI_COMM_WORLD);
    //MPI_Bcast(&m0, 1, MPI_DOUBLE,  0, MPI_COMM_WORLD);
    m0 = -0.1884;
    mass::m0 = m0;

    //The order in which these functions are called is very important

    LevelV::maxLevel = LevelV::levels-1;
    allocate_lattice_arrays(); 
    initializeMPI(); //2D rank topology
    readParameters("../inputs");
    boundaries();
    printParameters();

    srand((mpi::rank2d+1));
    
    spinor U(mpi::maxSizeH);
    spinor rhs(mpi::maxSizeH);
    spinor x0(mpi::maxSizeH); 
    for(int x = 1; x<=mpi::width_x; x++){
        for(int t = 1; t<=mpi::width_t; t++){
            int n = x*(mpi::width_t+2)+t;
            rhs.val[2*n]      = RandomU1();
            rhs.val[2*n+1]    = RandomU1();
            U.val[2*n] = RandomU1();
            U.val[2*n+1] = RandomU1();
        }
    }
    std::ostringstream NameData;
    int i = 0;
    double beta = 2;
    NameData << "../confs/2D_U1_Ns" << LV::Nx << "_Nt" << LV::Nt
                << "_b" << format(beta)
                << "_m" << format(mass::m0)
                << "_" << i << ".ctxt";
    read_binary(NameData.str(),U);
    double tol = 1e-10;
    Methods methods( U, rhs,  x0 ,m0,tol);
    //methods.BiCG(10000,true);
    //methods.CG(true);
    int m = 20, restarts = 1000; 
    //methods.GMRES(m,restarts,true);
    int xblocks = 4, tblocks = 4;
    //methods.SAP(500,xblocks,tblocks,true);
    //methods.FGMRES_sap(m,restarts,true);
    //methods.Vcycle(100,true);
    //methods.Kcycle(100,true);
    methods.FGMRES_amg_vcycle(AMGV::nu1,AMGV::nu2,true);

    methods.FGMRES_amg_kcycle(AMGV::nu1,AMGV::nu2,true);

    //test_open_conf();


 
    //Free coordinate arrays

    free_lattice_arrays();
    MPI_Finalize();

	return 0;
}