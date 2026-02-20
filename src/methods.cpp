#include "methods.h"

void Methods::BiCG(const int max_it, const bool print){   
    if (mpi::rank2d == 0) 
        std::cout << "--------------Bi-CGstab inversion--------------" << std::endl;
    BiCG::max_iter = max_it;
    BiCG::tol = tol;
    start = MPI_Wtime(); 
    bi_cgstab(U,rhs,x0,*xBiCG,m0,print);
    end = MPI_Wtime(); 
    if (mpi::rank2d == 0)
        printf("[rank %d] time elapsed Bi-CGstab: %.4fs.\n\n", mpi::rank2d, end - start);
}

void Methods::GMRES(const int len, const int restarts,const bool print){
    if (mpi::rank2d == 0)
        std::cout << "--------------GMRES without preconditioning--------------" << std::endl;
    int x_ini = 1, t_ini = 1, x_fin = mpi::width_x, t_fin = mpi::width_t;
    FGMRES_fine_level fgmres_fl(mpi::width_t*mpi::width_x, LV::dof, mpi::maxSizeH,
        x_ini, t_ini, 
        x_fin, t_fin, len, restarts, tol, U, m0);
    
    start = MPI_Wtime(); 
    fgmres_fl.fgmres(rhs,x0,*xGMRES,print);
    end = MPI_Wtime(); 
    if (mpi::rank2d == 0)
        printf("[rank %d] time elapsed GMRES: %.4fs.\n\n", mpi::rank2d, end - start);
}

void Methods::CG(const bool print){
    if (mpi::rank2d == 0)
        std::cout << "--------Inverting the normal equations with CG----------" << std::endl; 
    start = MPI_Wtime();
    conjugate_gradient(U, rhs, *xCG, m0,print);
    end = MPI_Wtime();

    if (mpi::rank2d == 0)
        printf("[rank %d] time elapsed CG: %.4fs.\n\n", mpi::rank2d, end - start);  
}

void Methods::SAP(const int iterations, const int xblocks, const int tblocks,const bool print){
    double tol = 1e-10;
    if (mpi::rank2d == 0)
        std::cout << "--------------SAP as stand-alone solver --------------" << std::endl;
    SAP_fine_level sap(mpi::width_x,  mpi::width_t, xblocks, tblocks, 2, 1);
    sap.set_params(U,mass::m0);
    start = MPI_Wtime();
    sap.SAP(rhs,*xSAP,iterations,tol,print);
    end = MPI_Wtime();
    if (mpi::rank2d == 0)
        printf("[rank %d] time elapsed SAP: %.4fs.\n\n", mpi::rank2d, end - start);
}


/*
void Methods::FGMRES_sap(spinor& x, const bool print){
    const bool save = false;
    int rank, size; 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        std::cout << "--------------Flexible GMRES with SAP preconditioning version --------------" << std::endl;
     
    MPI_Barrier(MPI_COMM_WORLD);
    FGMRES_SAP fgmres_sap(LV::Ntot, 2, FGMRESV::fgmres_restart_length, FGMRESV::fgmres_restarts,FGMRESV::fgmres_tolerance,GConf.Conf, m0);
    startT = MPI_Wtime();
    fgmres_sap.fgmres(rhs,x0,x,print,save);
    endT = MPI_Wtime();
    printf("[rank %d] time elapsed during FGMRES_SAP implementation: %.4fs.\n", rank, endT - startT);
    fflush(stdout);

}

void Methods::multigrid(spinor& x, const bool print){
    int rank, size; 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        std::cout << "--------------Stand-alone AMG --------------" << std::endl;
    startT = MPI_Wtime();
    AlgebraicMG AMG(GConf, m0,AMGV::nu1, AMGV::nu2);
    AMG.setUpPhase(AMGV::Nit);
    MPI_Barrier(MPI_COMM_WORLD);
    //AMG.testSetUp();
    AMG.applyMultilevel(100, rhs,x,1e-10,true);
    endT = MPI_Wtime();
    printf("[MPI process %d] time elapsed during the job: %.4fs.\n", rank, endT - startT);
}

void Methods::check_solution(const spinor& x_sol){
    spinor xini(LV::Ntot, c_vector(2, 0)); //Initial guess
    D_phi(GConf.Conf, x_sol, xini, m0); //D_phi U x
    for(int i = 0; i< LV::Ntot; i++){
        if (std::abs(xini[i][0] - rhs[i][0]) > 1e-8 || std::abs(xini[i][1] - rhs[i][1]) > 1e-8) {
            std::cout << "Solution not correct at index " << i << ": " << xini[i][0] << " != " << rhs[i][0] << " or " << xini[i][1] << " != " << rhs[i][1] << std::endl;
            return ;
        }
    }
    std::cout << "Solution is correct" << std::endl;
}
    */