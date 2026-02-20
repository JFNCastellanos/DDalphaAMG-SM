#include "methods.h"

void Methods::BiCG(spinor& x,const int max_it, const bool print){   
    const bool save = false;
    std::cout << "--------------Bi-CGstab inversion--------------" << std::endl;
    start = clock();
    x = bi_cgstab(&D_phi,LV::Ntot,2,GConf.Conf, rhs, x0, m0, max_it, 1e-10, print,save);
    end = clock();
    elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed time for Bi-CGstab = " << elapsed_time << " seconds" << std::endl;
}

void Methods::GMRES(spinor& x, const int len, const int restarts,const bool print){
    const bool save = false;
    std::cout << "--------------GMRES without preconditioning--------------" << std::endl;
    FGMRES_fine_level fgmres_fine_level(LV::Ntot, 2, len, restarts,1e-10,GConf.Conf, m0);
    start = clock();
    fgmres_fine_level.fgmres(rhs,x0,x,print,save);
    end = clock();
    elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed time for GMRES = " << elapsed_time << " seconds" << std::endl; 
}

void Methods::CG(spinor& x){
    std::cout << "--------Inverting the normal equations with CG----------" << std::endl; 
    start = clock();
    conjugate_gradient(GConf.Conf, rhs, x, m0);
    end = clock();
    elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed time for CG = " << elapsed_time << " seconds" << std::endl;  
}

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

void Methods::SAP(spinor& x,const int iterations, const bool print){
    int rank, size; 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        std::cout << "--------------SAP as stand-alone solver --------------" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    startT = MPI_Wtime();
    sap.SAP(rhs,x,iterations, SAPV::sap_blocks_per_proc,print);
    endT = MPI_Wtime();
    printf("[rank %d] time elapsed during SAP implementation: %.4fs.\n", rank, endT - startT);
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