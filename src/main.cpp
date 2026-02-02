#include <time.h> 
#include <ctime>
#include <fstream>
#include "conjugate_gradient.h"
#include "sap.h"
#include "tests.h"


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
        std::cout << "  -----------------------------" << std::endl;
        std::cout << "|    Halo exchange testing     |" << std::endl;
        std::cout << "  -----------------------------" << std::endl;
        std::cout << "Nx " << LV::Nx << " Nt " << LV::Nt << std::endl;
        std::cout << "ranks_x: ";
        std::cin >> mpi::ranks_x;
        std::cout << "ranks_t: ";
        std::cin >> mpi::ranks_t;
        std::cout << "m0: ";
        std::cin >> m0;
       
    }
    
    MPI_Bcast(&mpi::ranks_x, 1, MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&mpi::ranks_t, 1, MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&m0, 1, MPI_DOUBLE,  0, MPI_COMM_WORLD);
    
    initializeMPI(); //2D rank topology
    allocate_lattice_arrays(); //Allocates memory for arrays of coordinates
    boundaries();


    SAP_fine_level sap(mpi::width_x,  mpi::width_t, SAPV::sap_block_x, SAPV::sap_block_t, 2, 1);

    srand((mpi::rank2d+1));
       

    
    spinor U(mpi::maxSizeH);
    spinor phi(mpi::maxSizeH);
    spinor sol_SAP(mpi::maxSizeH);
    spinor sol_fgmres(mpi::maxSizeH);
    spinor sol_BiCG(mpi::maxSizeH);
    spinor sol_fgmresSAP(mpi::maxSizeH);
    spinor x0(mpi::maxSizeH);


    for(int x = 1; x<=mpi::width_x; x++){
        for(int t = 1; t<=mpi::width_t; t++){
            int n = x*(mpi::width_t+2)+t;
            U.val[2*n]      = RandomU1();
            U.val[2*n+1]    = RandomU1();
            phi.val[2*n]      = RandomU1();
            phi.val[2*n+1]    = RandomU1();
        }
    }

    sap.set_params(U, m0); //Setting gauge conf and m0 for SAP 

    bool message = true;
    int nu = 100;
    sap.SAP(phi,sol_SAP,nu, SAPV::sap_tolerance,  message);


    int m = 20;
    int restarts = 500;
    double tol = 1e-10;
    int x_ini = 1, t_ini = 1, x_fin = mpi::width_x, t_fin = mpi::width_t;
    FGMRES_fine_level fgmres_fl( LV::Ntot, LV::dof, mpi::maxSizeH,
        x_ini, t_ini, 
        x_fin, t_fin, m, restarts, tol, U, m0);

    FGMRES_SAP fgmres_sap(LV::Ntot, LV::dof, mpi::maxSizeH,
        x_ini, t_ini, 
        x_fin, t_fin, m, restarts, tol, U, m0); 

    bool print_message = true;

    if (mpi::rank2d == 0)
         std::cout << "----------GMRES----------" << std::endl;
    MPI_Barrier(mpi::cart_comm);
    fgmres_fl.fgmres(phi, x0, sol_fgmres, print_message);
    MPI_Barrier(mpi::cart_comm);
    if (mpi::rank2d == 0)
        std::cout << "----------BiCG----------" << std::endl;
    bi_cgstab(U, phi, x0, sol_BiCG, m0, print_message);
    MPI_Barrier(mpi::cart_comm);
    if (mpi::rank2d == 0)
        std::cout << "----------FGMFRES with SAP----------" << std::endl;
    fgmres_sap.fgmres(phi,x0,sol_fgmresSAP,print_message);


    check_sol(U,phi,sol_fgmresSAP,m0);
    
    /*
    for(int i = 0; i < mpi::size; i++) {
        MPI_Barrier(mpi::cart_comm);
        if (i == mpi::rank2d) {
            std::cout << "rank " << mpi::rank2d << std::endl;
            for(int x = 1; x<=mpi::width_x; x++){
                for(int t = 1; t<=mpi::width_t; t++){
                    int n = (x-1)*mpi::width_t + (t-1);
                    std::cout << "sol_BiCG        " <<  sol_BiCG.val[idx(x,t,0)] << "    " <<  sol_BiCG.val[idx(x,t,1)] << std::endl;
                    std::cout << "sol_fgmres      " <<  sol_fgmres.val[idx(x,t,0)] << "    " << sol_fgmres.val[idx(x,t,1)] << std::endl;
                    std::cout << "sol_SAP         " <<  sol_SAP.val[idx(x,t,0)] << "    " << sol_SAP.val[idx(x,t,1)] << std::endl;
                    std::cout << std::endl;
                }
                //std::cout << " " <<std::endl;
            } 
        }
    }
    */
   
    

    fflush(stdout);

//TODO 
/*
    Implement SAP:
      
*/

    //Free coordinate arrays
    free_lattice_arrays();
    MPI_Finalize();

	return 0;
}

