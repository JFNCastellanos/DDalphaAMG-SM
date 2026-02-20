#include "methods.h"

void Methods::BiCG(const int max_it, const bool print){   
    if (mpi::rank2d == 0) 
        std::cout << "--------------Bi-CGstab inversion--------------" << std::endl;
    BiCG::max_iter = max_it;
    BiCG::tol = tol;
    start = MPI_Wtime(); 
    bi_cgstab(U,rhs,x0,xBiCG,m0,print);
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
    fgmres_fl.fgmres(rhs,x0,xGMRES,print);
    end = MPI_Wtime(); 
    if (mpi::rank2d == 0)
        printf("[rank %d] time elapsed GMRES: %.4fs.\n\n", mpi::rank2d, end - start);
}

void Methods::CG(const bool print){
    if (mpi::rank2d == 0)
        std::cout << "--------Inverting the normal equations with CG----------" << std::endl; 
    start = MPI_Wtime();
    conjugate_gradient(U, rhs, xCG, m0,print);
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
    sap.SAP(rhs,xSAP,iterations,tol,print);
    end = MPI_Wtime();
    if (mpi::rank2d == 0)
        printf("[rank %d] time elapsed SAP: %.4fs.\n\n", mpi::rank2d, end - start);
}



void Methods::FGMRES_sap(const int len, const int restarts, const bool print){
    if (mpi::rank2d == 0)
        std::cout << "--------------Flexible GMRES with SAP preconditioning version --------------" << std::endl;
    
    int x_ini = 1, t_ini = 1, x_fin = mpi::width_x, t_fin = mpi::width_t;
    FGMRES_SAP fgmres_sap(LV::Ntot, LV::dof, mpi::maxSizeH,
    x_ini, t_ini, 
    x_fin, t_fin, len, restarts, tol, U, m0); 

    start = MPI_Wtime();
    fgmres_sap.fgmres(rhs,x0, xFGMRES_SAP,print);
    end = MPI_Wtime();

    if (mpi::rank2d == 0)
        printf("[rank %d] time elapsed FGMRES_SAP: %.4fs.\n\n", mpi::rank2d, end - start);
}

void Methods::Vcycle(const int iterations,const bool print){
    if (mpi::rank2d == 0)
        std::cout << "--------------Stand-alone AMG with a V-cycle----------------" << std::endl;

    AlgebraicMG AMG(U, m0,AMGV::nu1, AMGV::nu2);
    AMGV::cycle == 0; //V-cycle
    start = MPI_Wtime();
    AMG.setUpPhase(AMGV::Nit);
    AMG.applyMultilevel(iterations, rhs, xVcycle,tol,print);  
    end = MPI_Wtime();
    if (mpi::rank2d == 0)
        printf("[rank %d] time elapsed V-cycle AMG: %.4fs.\n\n", mpi::rank2d, end - start);
}

void Methods::Kcycle(const int iterations,const bool print){
    if (mpi::rank2d == 0)
        std::cout << "--------------Stand-alone AMG with a K-cycle----------------" << std::endl;

    AlgebraicMG AMG(U, m0,AMGV::nu1, AMGV::nu2);
    AMGV::cycle = 1; //K-cycle
    start = MPI_Wtime();
    AMG.setUpPhase(AMGV::Nit);
    AMG.applyMultilevel(iterations, rhs, xKcycle,tol,print);  
    end = MPI_Wtime();
    if (mpi::rank2d == 0)
        printf("[rank %d] time elapsed K-cycle AMG: %.4fs.\n\n", mpi::rank2d, end - start);
}

void Methods::FGMRES_amg(const int nu1, const int nu2,const int cycle,const bool print){
    if (mpi::rank2d == 0 && cycle == 0)
        std::cout << "--------------FGMRES with AMG V-cycle--------------" << std::endl;
    else if (mpi::rank2d == 0 && cycle == 1)
        std::cout << "--------------FGMRES with AMG K-cycle--------------" << std::endl;
    
    start = MPI_Wtime();
    FGMRES_AMG f_amg(U, FGMRESV::fgmres_restart_length, FGMRESV::fgmres_restarts, tol,nu1, nu2, cycle,m0);
    f_amg.fgmres(rhs,x0,xFGMRES_AMG,print);
    end = MPI_Wtime();
    
    if (mpi::rank2d == 0)
        printf("[rank %d] time elapsed FGMRES AMG: %.4fs.\n\n", mpi::rank2d, end - start);
}


/*
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