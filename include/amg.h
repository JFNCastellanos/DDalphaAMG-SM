#ifndef AMG_H
#define AMG_H

#include "level.h"

class AlgebraicMG{
    /*
	GaugeConf GConf: Gauge configuration
    m0: Mass parameter for the Dirac matrix
    nu1: Number of pre-smoothing iterations
    nu2: Number of post-smoothing iterations
	*/
public:

	//FGMRES for the k-cycle
   /* class FGMRES_k_cycle : public FGMRES {
	public:
    	FGMRES_k_cycle(const int& dim1, const int& dim2, const int& m, const int& restarts, const double& tol, AlgebraicMG* parent, int l) : 
		FGMRES(dim1, dim2, m, restarts, tol), parent(parent), l(l) {}
    
    	~FGMRES_k_cycle() { };
    
	private:
		AlgebraicMG* parent; //Pointer to the enclosing AMG instance
		int l; //Level

    	
    	//Implementation of the function that computes the matrix-vector product for the current level
    	
    	void func(const spinor& in, spinor& out) override {
        	parent->levels[l]->D_operator(in,out);
    	}
		//Preconditioning with the k-cycle
		void preconditioner(const spinor& in, spinor& out) override {
            parent->k_cycle(l,in,out); 
		}
	};
*/
    AlgebraicMG(const spinor & U, const double& m0, const int& nu1, const int& nu2) 
	: U(U), m0(m0), nu1(nu1), nu2(nu2){

    	for(int l = 0; l<LevelV::levels; l++){
        	Level* level = new Level(l,U);
        	levels.push_back(level);
			//We don't need this FGMRES for the coarsest level and the finest level
			/*FGMRES_k_cycle* fgmres = new FGMRES_k_cycle(LevelV::Nsites[l], 
				LevelV::DOF[l], 
				AMGV::fgmres_k_cycle_restart_length, 
				AMGV::fgmres_k_cycle_restarts, 
				AMGV::fgmres_k_cycle_tol, 
				this,
				l);
			fgmres_k_cycle_l.push_back(fgmres);
            */
    	}

    
    }    
    	
    ~AlgebraicMG() {
        for (auto ptr : levels) delete ptr;
       // for (auto ptr : fgmres_k_cycle_l) delete ptr;
    }
    //Pages 84 and 85 of Rottmann's thesis explain how to implement this ...
    void setUpPhase(const int& Nit);
//private:    
    const spinor U;
	double m0; 
	const int nu1; const int nu2; 
    std::vector<Level*> levels; //If I try to use a vector of objects I will run out of memory
	//std::vector<FGMRES_k_cycle*> fgmres_k_cycle_l; //Flexible GMRES used for the k-cycle on every level

    //Checks orthonormalization and verifies that P^+ D P = Dc
    //void testSetUp();
    //Checks that SAP is working properly
    //void testSAP();

    // psi_l = V_cycle(l,eta_l)
    //void v_cycle(const int& l, const spinor& eta_l, spinor& psi_l);

	// psi_l = K_cycle(l,eta_l)
	//void k_cycle(const int& l, const spinor& eta_l, spinor& psi_l);

    //Calls K or V-cycle depending on the value of AMGV::cycle. Stand-alone solver
    //void applyMultilevel(const int& it, const spinor&rhs, spinor& out,const double tol,const bool print_message);
    

};


//    FGMRES with a multilevel method as preconditioner
/*
class FGMRES_AMG : public FGMRES {
    public:
    FGMRES_AMG(const int& dim1, const int& dim2, const int& m, const int& restarts, const double& tol,
    const GaugeConf& GConf,const double& m0) : FGMRES(dim1, dim2, m, restarts, tol), GConf(GConf),
    m0(m0), dim1(dim1), dim2(dim2), amg(GConf, m0, AMGV::nu1, AMGV::nu2) {

    //      Set up phase for AMG     //
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double elapsed_time;
    double startT, endT;     
    startT = MPI_Wtime();
    amg.setUpPhase(AMGV::Nit); //test vectors intialization
    endT = MPI_Wtime();
    elapsed_time = endT - startT;
    std::cout << "[MPI Process " << rank << "] Elapsed time for Set-up phase = " << elapsed_time << " seconds" << std::endl;   
    //---------------------------//
    //Tests
    //amg.testSetUp(); //Checks that test vectors are orthonormal and that P^dagg D P = D_c at every level
    //amg.testSAP(); //Checks that SAP is working properly for every level. This compares the solution with GMRES.
    
    };
    ~FGMRES_AMG() { };
    
private:
    const GaugeConf& GConf; //Gauge configuration
    const double& m0; //reference to mass parameter
    const int &dim1;
    const int &dim2;
    int rank;
    AlgebraicMG amg; //AMG instance for the two-grid method
        
    void func(const spinor& in, spinor& out) override {
        D_phi(GConf.Conf, in, out, m0); 
    }

    void preconditioner(const spinor& in, spinor& out) override {
        for(int i = 0; i<dim1; i++){
            for(int j = 0; j<dim2; j++){
                out[i][j] = 0;
            }
        }
		if (AMGV::cycle == 0)
			amg.v_cycle(0,in, out);
		else if (AMGV::cycle == 1)
			amg.k_cycle(0,in, out);
    }
};
*/

#endif