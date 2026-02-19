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