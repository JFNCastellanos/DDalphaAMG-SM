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
        std::cout << "Test gather for G1 at level " << l << " with DOF=" << DOF*DOF << std::endl;
    test_gather_Datatypes_level_class(lev,DOF*DOF);
     if (mpi::rank2d == 0)
        std::cout << "Test gather for G2G3 at level " << l << " with DOF=" << DOF*DOF*2 << std::endl;
    test_gather_Datatypes_level_class(lev,DOF*DOF*2);
}


void test_scatter_Datatypes_level_class(Level& lev,const int dof){
    if (mpi::size != 16){
         printf("This test is meant to be run with 16 processes.\n");
        MPI_Abort(mpi::cart_comm, EXIT_FAILURE);
    }
    spinor input((lev.Nx_coarse_rank+2)*(lev.Nt_coarse_rank+2)*dof);
    spinor buffer((lev.Nt+2)*(lev.Nx+2)*dof);

	int commID = mpi::rank_dictionary[mpi::rank2d];
    int local_rank;
    MPI_Comm_rank(mpi::coarse_comm[commID], &local_rank);

    int n;
    for(int x = 1; x<=lev.Nx_coarse_rank; x++){
        for(int t = 1; t<=lev.Nt_coarse_rank; t++){
            n = x*(lev.Nt_coarse_rank+2)+t;
            for(int mu=0; mu<dof; mu++){
                input.val[dof*n+mu] = dof*n+mu + mpi::rank2d;
            }        
        }
    }


    for(int i = 0; i < mpi::size; i++) {
        MPI_Barrier(mpi::cart_comm);
        if (i == mpi::rank2d && local_rank == 0) {
            std::cout << "rank " << mpi::rank2d << std::endl;
            for(int x = 1; x<=lev.Nx_coarse_rank; x++){
                for(int t = 1; t<=lev.Nt_coarse_rank; t++){
                    int n = x*(lev.Nt_coarse_rank+2) + t;
                     std::cout << "[";
                    for(int mu=0; mu<dof;mu++){
                        std::cout << input.val[dof*n+mu] << ", ";
                    }
                    std::cout << "], ";
                }
                std::cout << std::endl;
            } 
        }
    }

    lev.scatter_to_local_rank_from_coarse_rank(input, buffer,dof);
    
    for(int i = 0; i <mpi::size; i++) {
        MPI_Barrier(mpi::cart_comm);
        if (i == mpi::rank2d) {
            std::cout << "\nScatter in rank " << mpi::rank2d << std::endl;
            for(int x = 0; x<lev.Nx+2; x++){
                for(int t = 0; t<lev.Nt+2; t++){
                    int n = x*(lev.Nt+2) + t;
                    std::cout << "[";
                    for(int mu=0; mu<dof;mu++){
                        std::cout << buffer.val[dof*n+mu] << ", ";
                    }
                    std::cout << "], ";
                }
                std::cout << std::endl;
            } 
        }
    }
}

void scatter_tests(){
    spinor U(mpi::maxSizeH);
    int l = 0;
    Level lev(l,U);
    int DOF = lev.DOF;
    if (mpi::rank2d == 0)
        std::cout << "Test scatter for spinor at level " << l << " with DOF=" << DOF << std::endl;
    test_scatter_Datatypes_level_class(lev,DOF);
     if (mpi::rank2d == 0)
        std::cout << "Test scatter for G1 at level " << l << " with DOF=" << DOF*DOF << std::endl;
    test_scatter_Datatypes_level_class(lev,DOF*DOF);
     if (mpi::rank2d == 0)
        std::cout << "Test scatter for G2G3 at level " << l << " with DOF=" << DOF*DOF*2 << std::endl;
    test_scatter_Datatypes_level_class(lev,DOF*DOF*2);
}


void test_Dc_with_rank_coarsening(){
    std::vector<Level*> levels;
    spinor U(mpi::maxSizeH);

    //if (mpi::size != 16){
    //    printf("This test is meant to be run with 4 processes.\n");
    //    MPI_Abort(mpi::cart_comm, EXIT_FAILURE);
    //}
    for(int x = 1; x<=mpi::width_x; x++){
        for(int t = 1; t<=mpi::width_t; t++){
            int n = x*(mpi::width_t+2)+t;
            U.val[2*n]        = RandomU1();
            U.val[2*n+1]      = RandomU1();
        }
    }

    for(int l = 0; l<LevelV::levels; l++){
        Level* level = new Level(l,U);
        levels.push_back(level);
    }


    //Random test vectors ... 
    static std::mt19937 randomInt(mpi::rank2d); 
	std::uniform_real_distribution<double> distribution(-1.0, 1.0); //mu, standard deviation
    for(int l=0; l<LevelV::maxLevel; l++){
        for(int cc = 0; cc < levels[l]->Ntest; cc++){
            for(int x=1; x<=levels[l]->Nx; x++){
            for(int t=1; t<=levels[l]->Nt; t++){
	        for(int c=0; c<levels[l]->colors; c++){
	        for(int s=0; s<2; s++){
                int indx 	= x*(levels[l]->Nt+2)*levels[l]->colors*2 + t*levels[l]->colors*2 + c*2 	+ s;
                levels[l]->tvec[cc].val[indx] = distribution(randomInt) + I_number * distribution(randomInt);            
            }
            }
            }
            }  
        }   
        levels[l]->orthonormalize();         //Orthonormalize test vectors
        levels[l]->checkOrthogonality();     //Checking orthogonality   
        levels[l]->makeCoarseLinks(*levels[l+1]);        //Make coarse links
    }

    
    
    for(int l = 0; l<LevelV::maxLevel; l++){
        if (mpi::rank2d == 0)
            std::cout << "checking level " << l+1 << std::endl;
        spinor vc((levels[l+1]->Nt+2)*(levels[l+1]->Nx+2)*levels[l+1]->DOF);
        spinor v((levels[l]->Nt_coarse_rank+2)*(levels[l]->Nx_coarse_rank+2)*levels[l]->DOF);
        spinor temp((levels[l]->Nt_coarse_rank+2)*(levels[l]->Nx_coarse_rank+2)*levels[l]->DOF);
        for(int x = 1; x<=levels[l+1]->Nx; x++){
        for(int t = 1; t<=levels[l+1]->Nt; t++){
        for(int dof=0; dof<levels[l+1]->DOF; dof++){
            int n = x*(levels[l+1]->Nt+2)+t;
            vc.val[levels[l+1]->DOF*n+dof] = RandomU1();
        }
        }
        }

        spinor out1((levels[l+1]->Nt_coarse_rank+2)*(levels[l+1]->Nx_coarse_rank+2)*levels[l+1]->DOF);
        spinor out2((levels[l]->tblocks_per_coarse_rank+2)*(levels[l]->xblocks_per_coarse_rank+2)*levels[l+1]->DOF);
        // Dc = P^+ D P
        levels[l+1]->D_operator(vc, out1);

        //Explicit application of each operator. Should coincide with D_operator at leve l+1.
        levels[l]->P_vc(vc,v);
        levels[l]->D_operator(v,temp);
        levels[l]->Pdagg_v(temp,out2);
        
        //Only check on the workinh ranks of the coarse level ...
        if (levels[l+1]->ranks_comm != MPI_COMM_NULL){
            for(int x = 1; x<=levels[l+1]->Nx; x++){
                for(int t = 1; t<=levels[l+1]->Nt; t++){
                    int n = x*(levels[l+1]->Nt+2)+t;
                    for(int dof=0; dof<levels[l+1]->DOF; dof++){
                        if (std::abs(out1.val[levels[l+1]->DOF*n+dof]-out2.val[levels[l+1]->DOF*n+dof]) > 1e-8){
                            std::cout << "Both implementations of D don't coincide, rank " << mpi::rank2d << " level " << l+1 << std::endl;
                            std::cout << "x " << x << " t " << t << " dof " << dof << std::endl;
                            std::cout << "D_operator vc      " <<  out1.val[levels[l+1]->DOF*n+dof] << std::endl;
                            std::cout << "P^+ D P vc         " <<  out2.val[levels[l+1]->DOF*n+dof] << std::endl;
                            //std::cout << "vc                 " <<  vc.val[levels[l+1]->DOF*n+dof] << std::endl;
                            std::cout << std::endl;      
                            return; 
                        }
                    } 
                }
            }
            std::cout << "Dc coincides with P^+ D P at level " << l+1  << " on rank " << mpi::rank2d << std::endl;
        }
            
        
        
    }
    
        

    for (auto ptr : levels) delete ptr;

}


void test_SAP_in_level(){   
    std::vector<Level*> levels;
    spinor U(mpi::maxSizeH);

    //if (mpi::size != 16){
    //    printf("This test is meant to be run with 4 processes.\n");
    //    MPI_Abort(mpi::cart_comm, EXIT_FAILURE);
    //}
    for(int x = 1; x<=mpi::width_x; x++){
        for(int t = 1; t<=mpi::width_t; t++){
            int n = x*(mpi::width_t+2)+t;
            U.val[2*n]        = RandomU1();
            U.val[2*n+1]      = RandomU1();
        }
    }

    for(int l = 0; l<LevelV::levels; l++){
        Level* level = new Level(l,U);
        levels.push_back(level);
    }

    //Random test vectors ... 
    static std::mt19937 randomInt(mpi::rank2d); 
	std::uniform_real_distribution<double> distribution(-1.0, 1.0); //mu, standard deviation
    for(int l=0; l<LevelV::maxLevel; l++){
        for(int cc = 0; cc < levels[l]->Ntest; cc++){
            for(int x=1; x<=levels[l]->Nx; x++){
            for(int t=1; t<=levels[l]->Nt; t++){
	        for(int c=0; c<levels[l]->colors; c++){
	        for(int s=0; s<2; s++){
                int indx 	= x*(levels[l]->Nt+2)*levels[l]->colors*2 + t*levels[l]->colors*2 + c*2 	+ s;
                levels[l]->tvec[cc].val[indx] = distribution(randomInt) + I_number * distribution(randomInt);            
            }
            }
            }
            }  
        }   
        levels[l]->orthonormalize();                //Orthonormalize test vectors
        levels[l]->checkOrthogonality();            //Checking orthogonality   
        levels[l]->makeCoarseLinks(*levels[l+1]);   //Make coarse links
    }


    //Checking that sap_l gives the same result as sap_fine_level
    int l = 0;
    spinor rhs((levels[l]->Nt+2)*(levels[l]->Nx+2)*levels[l]->DOF);
    spinor x_level((levels[l]->Nt+2)*(levels[l]->Nx+2)*levels[l]->DOF);

    spinor Dphi_level((levels[l]->Nt+2)*(levels[l]->Nx+2)*levels[l]->DOF);
    spinor Dphi((levels[l]->Nt+2)*(levels[l]->Nx+2)*levels[l]->DOF);
    for(int x=1; x<=levels[l]->Nx; x++){
    for(int t=1; t<=levels[l]->Nt; t++){
	for(int dof=0; dof<levels[l]->DOF; dof++){
        int indx= (x*(levels[l]->Nt+2)+t)*levels[l]->DOF + dof;
        rhs.val[indx] = 1;
    }
    }
    }
    
    spinor x_fine((levels[l]->Nt+2)*(levels[l]->Nx+2)*levels[l]->DOF);
    SAP_fine_level sap(mpi::width_x,  mpi::width_t, LevelV::SAP_Block_x[l], LevelV::SAP_Block_t[l], 2, 1);
    sap.set_params(U,mass::m0);


    /*
    int block = 0;
    spinor in(sap.Nvars_w_halo);
    for(int x=1; x<=sap.x_elements; x++){
    for(int t=1; t<=sap.t_elements; t++){
	for(int dof=0; dof<sap.dofs; dof++){
        int indx= (x*(sap.t_elements+2)+t)*sap.dofs + dof;
        in.val[indx] = 1;
    }
    }
    }

    spinor out(sap.Nvars_w_halo);
    spinor out_level(sap.Nvars_w_halo);

    sap.D_B(U,in,out,mass::m0,block);
    levels[l]->sap_l->D_local(in,out_level,block);

    for(int x=1; x<=sap.x_elements; x++){
    for(int t=1; t<=sap.t_elements; t++){
	for(int dof=0; dof<sap.dofs; dof++){
        int indx= (x*(sap.t_elements+2)+t)*sap.dofs + dof;
         if (std::abs(out_level.val[indx]-out.val[indx]) > 1e-8){
            std::cout << "Different solutions on rank " << mpi::rank2d << " at level " << l << std::endl;
            std::cout << "out_level " << out_level.val[indx] << std::endl;
            std::cout << "out       " << out.val[indx]  << std::endl;
            std::cout << "x " << x << " t " << t << " dof " << std::endl;
           // return ;
         }
    }
    }
    }
    */

    double tol=1e-10;
    bool print=true;
    int nu = 100;
/*
    D_phi(U, rhs,  Dphi,mass::m0);

    levels[l]->D_operator(rhs, Dphi_level);

    
    for(int x=1; x<=levels[l]->Nx; x++){
    for(int t=1; t<=levels[l]->Nt; t++){
	for(int dof=0; dof<levels[l]->DOF; dof++){
        int indx= (x*(levels[l]->Nt+2)+t)*levels[l]->DOF + dof;
         if (std::abs(Dphi_level.val[indx]-Dphi.val[indx]) > 1e-8){
            std::cout << "Different solutions on rank " << mpi::rank2d << " at level " << l << std::endl;
            std::cout << "Dphi_level " << Dphi_level.val[indx] << std::endl;
            std::cout << "Dphi       " << Dphi.val[indx]  << std::endl;
            return ;
         }
    }
    }
    }
*/


   
    if (mpi::rank2d == 0)
        std::cout << "SAP solver inside class Level" << std::endl;
    levels[l]->sap_l->SAP(rhs,x_level,nu,tol,print);

    if (mpi::rank2d == 0)
        std::cout << "Outer SAP solver" << std::endl;
   
    sap.SAP(rhs,x_fine,nu,tol,print);
    

    

    
    for(int x=1; x<=levels[l]->Nx; x++){
    for(int t=1; t<=levels[l]->Nt; t++){
	for(int dof=0; dof<levels[l]->DOF; dof++){
        int indx= (x*(levels[l]->Nt+2)+t)*levels[l]->DOF + dof;
         if (std::abs(x_level.val[indx]-x_fine.val[indx]) > 1e-8){
            std::cout << "Different solutions on rank " << mpi::rank2d << " at level " << l << std::endl;
            std::cout << "x_level " << x_level.val[indx] << std::endl;
            std::cout << "x_fine  " << x_fine.val[indx]  << std::endl;
            return ;
         }
    }
    }
    }
 
    std::cout << "Both implementations give the same solution on rank " << mpi::rank2d << std::endl;

}