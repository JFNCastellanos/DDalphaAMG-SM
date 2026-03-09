#include <time.h> 
#include <ctime>
#include "conjugate_gradient.h"
#include "sap.h"
#include "params.h"
#include "methods.h"
#include "io.h"
#include "tests.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi::size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi::rank);
        
    //srand(mpi::rank*time(0));
    std::string confFile;
    std::string rhsFile;
    std::string pFile;
    
    
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
        std::cout << "m0: ";
        std::cin >> m0;
        std::cout << "Configuration file path: ";
        std::cin >> confFile;
        std::cout << "RHS file path: ";
        std::cin >> rhsFile;
        std::cout << "Method parameters file path: ";
        std::cin >> pFile;
    }

    MPI_Bcast(&mpi::ranks_x, 1, MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&mpi::ranks_t, 1, MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&LevelV::levels, 1, MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&m0, 1, MPI_DOUBLE,  0, MPI_COMM_WORLD);
    broadcast_file_name(confFile);
    broadcast_file_name(rhsFile);
    broadcast_file_name(pFile);
    mass::m0 = m0;

    
    //The order in which these functions are called is very important
    LevelV::maxLevel = LevelV::levels-1;
    allocate_lattice_arrays(); 
    initializeMPI(); //2D rank topology
    readParameters(pFile);
    boundaries();
    printParameters();
    //--------------------------------------//

    srand((mpi::rank2d+1));
    
    spinor U(mpi::maxSizeH);
    spinor rhs(mpi::maxSizeH);
    spinor x0(mpi::maxSizeH);   //Zero vector as initial solution

    read_binary(confFile,U);
    read_binary(rhsFile,rhs);
    
    double tol = 1e-10;
    Methods methods(U,rhs,x0,m0,tol);
    methods.BiCG(10000,true);
    methods.CG(true);
    int m = 20, restarts = 1000; 
    methods.GMRES(m,restarts,true);
    int xblocks = 4, tblocks = 4;
    methods.SAP(100,xblocks,tblocks,true);
    methods.FGMRES_sap(m,restarts,true);
    //methods.Vcycle(100,true);
    //methods.Kcycle(100,true);
    methods.FGMRES_amg_vcycle(AMGV::nu1,AMGV::nu2,true);
    methods.FGMRES_amg_kcycle(AMGV::nu1,AMGV::nu2,true);
    if (mpi::rank2d == 0)
        std::cout << "Checking solution of V-cycle" << std::endl;
    methods.check_solution(methods.xFGMRES_AMG_vcycle);

    if (mpi::rank2d == 0)
        std::cout << "Checking solution of K-cycle" << std::endl;
    methods.check_solution(methods.xFGMRES_AMG_kcycle);
 
    //Free coordinate arrays

    free_lattice_arrays();
    MPI_Finalize();

	return 0;
}