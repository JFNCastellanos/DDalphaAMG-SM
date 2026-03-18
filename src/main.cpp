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
        
    int nrhs;
    //To call the sequential program one has to choose ranks_x = ranks_t = 1
    if (mpi::rank == 0){
         //---Input data---//
        std::cout << " -----------------------------" << std::endl;
        std::cout << "| Right-hand sides generator |" << std::endl;
        std::cout << " -----------------------------" << std::endl;
        std::cout << "Nx " << LV::Nx << " Nt " << LV::Nt << std::endl;
        std::cout << "ranks_x: ";
        std::cin >> mpi::ranks_x;
        std::cout << "ranks_t: ";
        std::cin >> mpi::ranks_t;
        std::cout << "number of rhs to generate: ";
        std::cin >> nrhs;
    }

    MPI_Bcast(&mpi::ranks_x, 1, MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&mpi::ranks_t, 1, MPI_INT,  0, MPI_COMM_WORLD);

 
    allocate_lattice_arrays(); 
    initializeMPI(); //2D rank topology
    boundaries();
    //--------------------------------------//


    static std::mt19937 randomInt(mpi::rank2d); 
	std::uniform_real_distribution<double> distribution(-1.0, 1.0); //mu, standard deviation

    if (mpi::rank2d == 0){
        for(int i=0;i<nrhs;i++){
            spinor rhs(2*LV::Nx*LV::Nt);
            for (int x = 0; x < LV::Nx; x++) {
            for (int t = 0; t < LV::Nt; t++) {
                int n = x * LV::Nt + t;
                for (int mu = 0; mu < 2; mu++) {
                    rhs.val[2*n+mu] = distribution(randomInt) + I_number * distribution(randomInt);
                }
            }
            }
            save_rhs(i,rhs);
            //MPI_Barrier(mpi::cart_comm);
            //check_rhs(i,rhs);
        }
    }
     
    free_lattice_arrays();
    MPI_Finalize();

	return 0;
}


