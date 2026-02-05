#ifndef LEVEL_H
#define LEVEL_H

#include <algorithm>
#include "sap.h"

/*
    One level of the AMG method
*/
class Level {
public:   
    //SAP for smoothing D_operator. It is not used for the coarsest level but we declare it anyway.
    //-------------------------------Nested class-------------------------------//
    /*
    class SAP_level_l : public SAP_C {
    public:
        SAP_level_l(const int& dim1, const int& dim2, const double& tol,const int& Nt, const int& Nx,const int& block_x,const int& block_t,
        const int& spins, const int& colors,Level* parent) :
        SAP_C(dim1, dim2, tol, Nt, Nx, block_x, block_t,spins,colors), parent(parent) {        
        }

    private: 
        Level* parent; //Parent class

        
        //Global D operation
        
        void funcGlobal(const spinor& in, spinor& out) override { 
            parent->D_operator(in, out); //Dirac operator at the current level
        }

        
        //Local D operations
        
        void D_local(const spinor& in, spinor& out, const int& block);

        void funcLocal(const spinor& in, spinor& out) override { 
            D_local( in, out,current_block);
        }

        
        //    Given a lattice point with index n, it returns the corresponding 
        //    SAP block and the local index m within that block.
        
        inline void getMandBlock(const int& n, int &m, int &block) {
            int x = n / Nx; //x coordinate of the lattice point 
            int t = n % Nt; //t coordinate of the lattice point
            //Reconstructing the block and m index from x and t
            int block_x = x / x_elements; //Block index in the x direction
            int block_t = t / t_elements; //Block index in the t direction
            block = block_x * Block_t + block_t; //Block index in the SAP method

            int mx = x % x_elements; //x coordinate in the block
            int mt = t % t_elements; //t coordinate in the block
            m = mx * t_elements + mt; //Index in the block
        }

    };

    SAP_level_l sap_l; 
    */
    //----------------------------------------------------------------------------//
    //GMRES for the current level. We use it for solving the coarsest system. We could use it as as smoother as well.
    /*
    class GMRES_level_l : public FGMRES {
	public:
    	GMRES_level_l(const int& dim1, const int& dim2, const int& m, const int& restarts, const double& tol, Level* parent) : 
		FGMRES(dim1, dim2, m, restarts, tol), parent(parent) {}
    
    	~GMRES_level_l() { };
    
	private:
		Level* parent; //Pointer to the enclosing AMG instance
    	
    	//Implementation of the function that computes the matrix-vector product for the fine level
    	
    	void func(const spinor& in, spinor& out) override {
        	parent->D_operator(in,out);
    	}
		//No preconditioning for the coarsest level
		void preconditioner(const spinor& in, spinor& out) override {
            out = std::move(in); //Identity operation
		}
	};

	GMRES_level_l gmres_l;
    */
    //----------------------------------------------------------------------------//
    
    //Level Constructor
    Level(const int& level, const spinor& U) : level(level), U(U),
        Nx(LevelV::NxSites[level]/LevelV::RanksX[level]),
        Nt(LevelV::NtSites[level]/LevelV::RanksT[level]),
        Ntot(Nx*Nt)
    {

        //Test vectors
        tvec        = spinor(LevelV::Ntest[level]*Ntot*LevelV::DOF[level]);
        tvec_copy   = spinor(LevelV::Ntest[level]*Ntot*LevelV::DOF[level]);

         	    
        //Gauge links to define D_operator (matrix problem at this level). We define them with halos.
        int Ntot_halo = (Nx+2)*(Nt+2);
        G1 = spinor(Ntot_halo*2*2*colors*colors);
        G2 = spinor(Ntot_halo*2*2*colors*colors*2);
        G3 = spinor(Ntot_halo*2*2*colors*colors*2);

        exchange_halo(U.val); //We only exchange the halo once, the gauge field is not modified.

        if (level == 0)
            makeDirac(); 
        
        
    };

    ~Level() {}

    const spinor U; //Gauge configuration

    spinor tvec;        //[Ntest][Nt.Nx][colors.spins]
    spinor tvec_copy;

    const int level; 
    const int xblocks_per_rank = LevelV::BlocksX[level]/LevelV::RanksX[level]; //Number of blocks on x inside the current rank
    const int tblocks_per_rank = LevelV::BlocksT[level]/LevelV::RanksT[level]; //Number of blocks on t inside the current rank
    const int blocks_per_rank = xblocks_per_rank*tblocks_per_rank; //Number of blocks inside the rank
    const int Nx;   //Nx on the fine grid in rank r (no halo)
    const int Nt;   //Nt on the fine grid in rank r (no halo)
    const int Ntot; //Nx*Nt
    const int colors = LevelV::Colors[level];   //Number of colors at this level
    const int Ntest = (level != LevelV::maxLevel) ? LevelV::Ntest[level]: 1;     //Number of test vectors to go to the next level
    const int DOF = 2*colors;         //Degrees of freedom at each lattice site at this level

    const int x_elements = (level != LevelV::maxLevel) ?  Nx / xblocks_per_rank: 1;
    const int t_elements = (level != LevelV::maxLevel) ?  Nt / tblocks_per_rank: 1; 
    const int sites_per_block = x_elements * t_elements;
    const int NBlocks = (level != LevelV::maxLevel) ? LevelV::NBlocks[level]: 1; //Number of lattice blocks 
    
    //At level = 0 these vectors represent the gauge links.
    //At level > 1 they are the coarse gauge links generated in the previous level
    spinor G1; 
    spinor G2; 
    spinor G3; 


/*
	Prolongation operator times a spinor x = P v
	x_i = P_ij v_j. dim(P) = DOF Nsites x Ntest Nagg, 
	dim(v) = [NBlocks][2*Ntest], dim(x) = [Nsites][DOF]
    */
    void P_vc(const spinor& v,spinor& out);

    /*
	Restriction operator times a spinor on the coarse grid, x = P^H v
	x_i = P^dagg_ij v_j. dim(P^dagg) =  Ntest Nagg x DOF Nsites,
	dim(v) = [Nsites][DOF], dim(x) = [NBlocks][2*Ntest] 
    */
    void Pdagg_v(const spinor& v,spinor& out);

    void orthonormalize();      //Local orthonormalization of the test vectors
    void checkOrthogonality();  //Check orthogonality of the test vectors

    //Index functions for gauge links. These correspond to the current level
	//get index for A_coeff 1D array
    //[A(x)]^{alf,bet}_{c,b} --> A_coeff[x][alf][bet][c][b]
	inline int getG1index(const int& x, const int& alf, const int& bet, const int& c, const int& b){
		return x * 2 * 2 * colors * colors 
        + alf * 2 * colors * colors 
        + bet * colors * colors
        + c * colors 
        + b;
	}
	//[B_mu(x)]^{alf,bet}_{c,b}  --> B_coeff[x][alf][bet][c][b][mu]
    //[C_mu(x)]^{alf,bet}_{c,b}  --> C_coeff[x][alf][bet][c][b][mu]
	inline int getG2G3index(const int& x, const int& alf, const int& bet, const int& c, const int& b, const int& mu){
        return x * 2 * 2 * colors * colors * 2 
        + alf * 2 * colors * colors * 2 
        + bet * colors * colors * 2
        + c * colors * 2 
        + b * 2 
        + mu;
    }
    	
    //Index functions for coarse gauge links. These correspond to the next level, but are generated here (not stored)
	//get index for A_coeff 1D array
    //[A(x)]^{alf,bet}_{c,b} --> A_coeff[x][alf][bet][c][b]
	inline int getAindex(const int& block, const int& alf, const int& bet, const int& c, const int& b){
		return block * 2 * 2 * Ntest * Ntest 
        + alf * 2 * Ntest * Ntest 
        + bet * Ntest * Ntest
        + c * Ntest 
        + b;
	}
	//[B_mu(x)]^{alf,bet}_{c,b}  --> B_coeff[x][alf][bet][c][b][mu]
    //[C_mu(x)]^{alf,bet}_{c,b}  --> C_coeff[x][alf][bet][c][b][mu]
	inline int getBCindex(const int& block, const int& alf, const int& bet, const int& c, const int& b, const int& mu){
        return block * 2 * 2 * Ntest * Ntest * 2 
        + alf * 2 * Ntest * Ntest * 2 
        + bet * Ntest * Ntest * 2
        + c * Ntest * 2 
        + b * 2 
        + mu;
    }

    
    //Creates G1, G2 and G3
    void makeDirac();

    //Make coarse gauge links. They will be used in the next level as G1, G2 and G3.
    //void makeCoarseLinks(Level& next_level);//& A_coeff,c_vector& B_coeff, c_vector& C_coeff);


    /*
    Matrix-vector operation that defines the level l.
    For instance, at level = 0, D_operator is just the Dirac operator
    at level = 1 D_operator is Dc
    at level = 2 D_operator is (Dc)_c ...
    */
    void D_operator(const spinor& v, spinor& out);

};

#endif