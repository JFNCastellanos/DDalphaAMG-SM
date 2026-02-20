#include "amg.h"

void AlgebraicMG::setUpPhase(const int& Nit){
	
	static std::mt19937 randomInt(mpi::rank2d); //Same seed for all the MPI copies
	std::uniform_real_distribution<double> distribution(-1.0, 1.0); //mu, standard deviation
	

	//Generate test vectors at the fine level
    int indx;
	for (int cc = 0; cc < levels[0]->Ntest; cc++) {
	    for (int x = 1; x <= levels[0]->Nx; x++) {
        for (int t = 1; t <= levels[0]->Nt; t++) {
	    for (int dof = 0; dof < levels[0]->DOF; dof++) {
            indx = (x*(levels[0]->Nt+2)+t)*levels[0]->DOF+dof;
			levels[0]->tvec[cc].val[indx] = distribution(randomInt) + I_number * distribution(randomInt);
	    }
	    }
	    }
    }
	
	//v_l = P^dagger v_{l-1}
	for(int l=1; l<LevelV::levels-1; l++){
		for(int cc = 0; cc<levels[l]->Ntest;cc++){
			if (cc<levels[l-1]->Ntest){
				levels[l-1]->Pdagg_v(levels[l-1]->tvec[cc],levels[l]->tvec[cc]);
            }
			else{
				for (int x = 1; x <= levels[l]->Nx; x++) {
                for (int t = 1; t <= levels[l]->Nt; t++) {
	            for (int dof = 0; dof < levels[l]->DOF; dof++) {
                    indx = (x*(levels[l]->Nt+2)+t)*levels[l]->DOF+dof;
			        levels[l]->tvec[cc].val[indx] = distribution(randomInt) + I_number * distribution(randomInt);
	            }
	            }
	            }
			}
		}
	}

	//Smoothing the test vectors
    for(int l=0; l<LevelV::levels-1; l++){
        spinor rhs((levels[l]->Nt+2)*(levels[l]->Nx+2)*levels[l]->DOF);
		for (int cc = 0; cc <levels[l]->Ntest; cc++) {
			//Approximately solving D x = 0 with AMGV::SAP_test_vectors_iterations. Tolerance is irrelevant here.
            bool print = false; double tol=1e-10;
            levels[l]->sap_l->SAP(rhs,levels[l]->tvec[cc],AMGV::SAP_test_vectors_iterations,tol,print); //D^-1 rhs
		}
		levels[l]->orthonormalize(); 
		levels[l]->makeCoarseLinks(*levels[l+1]); 
	}

    if (mpi::rank2d == 0)std::cout << "Set-up phase finished" << std::endl;
	
}

void AlgebraicMG::v_cycle(const int& l, const spinor& eta_l, spinor& psi_l){
    double tol = 1e-10;
    int indx;
	if (l == LevelV::maxLevel){
		//For the coarsest level we use GMRES to find a solution
		levels[l]->gmres_l->fgmres(eta_l, eta_l, psi_l, false);                         //psi_l = D_l^-1 eta_l 
	}
	else{
		//Buffers
        spinor Dpsi((levels[l]->Nt+2)*(levels[l]->Nx+2)*(levels[l]->DOF));              //D_l psi_l
		spinor r_l((levels[l]->Nt+2)*(levels[l]->Nx+2)*(levels[l]->DOF));               //r_l = eta_l - D_l psi_l
		spinor eta_l_1((levels[l+1]->Nt+2)*(levels[l+1]->Nx+2)*(levels[l+1]->DOF));     //eta_{l+1}
		spinor psi_l_1((levels[l+1]->Nt+2)*(levels[l+1]->Nx+2)*(levels[l+1]->DOF));     //psi_{l+1}
		spinor P_psi((levels[l]->Nt+2)*(levels[l]->Nx+2)*(levels[l]->DOF));             //P_l psi_{l+1}

		//Pre - smoothing
		if (nu1 > 0)
			levels[l]->sap_l->SAP(eta_l,psi_l,nu1,tol,false); 
		
		//Coarse grid correction 
		levels[l]->D_operator(psi_l,Dpsi); 
        for (int x = 1; x <= levels[l]->Nx; x++) {
        for (int t = 1; t <= levels[l]->Nt; t++) {
	    for (int dof = 0; dof < levels[l]->DOF; dof++) {
            indx = (x*(levels[l]->Nt+2)+t)*levels[l]->DOF+dof;
			r_l.val[indx] = eta_l.val[indx] - Dpsi.val[indx];                           //r_l = eta_l - D_l psi_l
		}
		}
        }
		levels[l]->Pdagg_v(r_l,eta_l_1);                                                //eta_{l+1} = P^H (eta_l - D_l psi_l)
		v_cycle(l+1,eta_l_1,psi_l_1);                                                   //psi_{l+1} = V-Cycle(l+1,eta_{l+1})

		levels[l]->P_vc(psi_l_1,P_psi);                                                 //P_psi = P_l psi_{l+1}

		for (int x = 1; x <= levels[l]->Nx; x++) {
        for (int t = 1; t <= levels[l]->Nt; t++) {
	    for (int dof = 0; dof < levels[l]->DOF; dof++) {
            indx = (x*(levels[l]->Nt+2)+t)*levels[l]->DOF+dof;
			psi_l.val[indx] += P_psi.val[indx];                                             //psi_l = psi_l + P_l psi_{l+1}
		}
		}
        }

		//Post - smoothing
		if (AMGV::nu2 > 0)
			levels[l]->sap_l->SAP(eta_l,psi_l,nu2,tol,false); 
		
	}	

}




void AlgebraicMG::testSetUp(){
    for(int l = 0; l<LevelV::maxLevel; l++){
        if (mpi::rank2d == 0)
            std::cout << "Checking level " << l+1 << std::endl;
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
}

void AlgebraicMG::testSAP(){
    for(int l = 0; l<LevelV::levels; l++){
        //for(int l = 0; l<1; l++){
        if (levels[l]->ranks_comm != MPI_COMM_NULL){
            spinor rhs((levels[l]->Nt+2)*(levels[l]->Nx+2)*levels[l]->DOF);
            spinor x_level((levels[l]->Nt+2)*(levels[l]->Nx+2)*levels[l]->DOF);
            spinor D_x_level((levels[l]->Nt+2)*(levels[l]->Nx+2)*levels[l]->DOF);

            for(int x=1; x<=levels[l]->Nx; x++){
            for(int t=1; t<=levels[l]->Nt; t++){
	        for(int dof=0; dof<levels[l]->DOF; dof++){
                int indx= (x*(levels[l]->Nt+2)+t)*levels[l]->DOF + dof;
                rhs.val[indx] = RandomU1();
            }
            }
            }
    
   
            double tol=1e-10;
            bool print=true;
            int nu = 100;
   
            levels[l]->sap_l->SAP(rhs,x_level,nu,tol,print); //D^-1 rhs
        
            levels[l]->D_operator(x_level,D_x_level);//D D^-1 rhs

            for(int x=1; x<=levels[l]->Nx; x++){
            for(int t=1; t<=levels[l]->Nt; t++){
	        for(int dof=0; dof<levels[l]->DOF; dof++){
                int indx= (x*(levels[l]->Nt+2)+t)*levels[l]->DOF + dof;
                if (std::abs(D_x_level.val[indx]-rhs.val[indx]) > 1e-8){
                    std::cout << "rhs /= D_operator (D^-1 rhs) on rank " << mpi::rank2d << " at level " << l << std::endl;
                    std::cout << "D_x_level " << D_x_level.val[indx] << std::endl;
                    std::cout << "rhs       " << rhs.val[indx]  << std::endl;
                    return ;
                }
            }
            }
            }
 
            if (mpi::rank2d == 0)
                std::cout << "D_operator (SAP_l rhs) = rhs on rank " << mpi::rank2d << " level " << l << " i.e. solution is correct" << std::endl;
        }
    }
}