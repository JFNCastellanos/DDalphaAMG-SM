#include "dirac_operator.h"
#include "level.h"
#include "sap.h"

void check_sol(const spinor& U,const spinor& rhs, const spinor& inverse, const double& m0){
    spinor x(mpi::maxSizeH);
    D_phi(U,inverse,x,m0);
    int index;
    for (int nx=1;nx<=mpi::width_x;nx++){
    for(int nt=1;nt<=mpi::width_x;nt++){
    for(int mu=0; mu<2; mu++){
        index = idx(nx,nt,mu);
        if (std::abs(x.val[index]-rhs.val[index]) > 1e-8 ){
            std::cout << "Solution differs at nx" << nx << " nt " << nt << " mu " << mu << " rank" << mpi::rank2d << std::endl;
            std::cout << x.val[index] << "   " << rhs.val[index] << std::endl;
        }
    }
    }
    }

    MPI_Barrier(mpi::cart_comm); 
    if (mpi::rank2d == 0)
        std::cout << "All good with the solution!" << std::endl;
}


    
void test_level_l(const int& l, const spinor& U){
    Level level(l,U);
    int indxtv, indx;
    int Nt, Nx, colors, Ntest;
    Nt = level.Nt; Nx = level.Nx; colors = level.colors; Ntest = level.Ntest;
    int count = 0;
    for(int cc = 0; cc < level.Ntest; cc++){
        for(int t=0; t<level.Nt; t++){
		for(int x=0; x<level.Nx; x++){
		for(int c=0; c<level.colors; c++){
		for(int s=0; s<2; s++){
            indx 	= x*Nt*colors*2 + t*colors*2 + c*2 	+ s;
			indxtv 	= indx*Ntest + cc;
            level.tvec.val[indxtv] = count++;
        }
        }
        }
        }      
    }

    spinor ev(level.blocks_per_rank*2*Ntest); //Lives on the coarse lattice
    spinor column(Nx*Nt*2*colors);
    if (mpi::rank2d == 0){
        for(int i = 0; i < level.blocks_per_rank*2*Ntest; i++){
            ev.val[i] = 1;
            level.P_vc(ev,column);
            ev.val[i] = 0;
            for(int j = 0; j < Nx*Nt*2*colors; j++){
                std::cout << ev.val[j] << " ";
            }
            std::cout << std::endl;
        }
    }
    
}