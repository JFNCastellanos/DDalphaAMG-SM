#include "dirac_operator.h"
#include "level.h"
#include "sap.h"
#include <random>

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


    
//Assemble P and Pdagg for level l.
void AssembleP_Pdagg(const int& l, const spinor& U){
    Level level(l,U);
    int indxtv, indx;
    int Nt, Nx, colors, Ntest;
    Nt = level.Nt; Nx = level.Nx; colors = level.colors; Ntest = level.Ntest;
    
    for(int cc = 0; cc < level.Ntest; cc++){
        for(int x=0; x<level.Nx; x++){
        for(int t=0; t<level.Nt; t++){
	    for(int c=0; c<level.colors; c++){
	    for(int s=0; s<2; s++){
            indx 	= x*Nt*colors*2 + t*colors*2 + c*2 	+ s;
			indxtv 	= indx*Ntest + cc;
            level.tvec.val[indxtv] = indxtv+1;
           /* if (mpi::rank2d == 0){
                std::cout << "(x,t,c,s) = " << x << ", " << t << ", " << c << ", " << s <<  std::endl;
                std::cout << "indx " << indxtv << std::endl;
            }
            */
            
        }
    }
    }
    }      
    }

    if (mpi::rank2d == 0){
        std::cout << "Rank " << mpi::rank2d << std::endl;
        std::cout << "Nx " << Nx << " Nt " << Nt << " colors " << colors << " Ntest " << Ntest << std::endl;
        std::cout << "Blocks per rank " << level.blocks_per_rank << std::endl;
        std::cout << "count " << indxtv-1 << std::endl;
    
    }

    
    if (mpi::rank2d == 0){
        spinor ev(level.blocks_per_rank*2*Ntest); //Lives on the coarse lattice
        spinor column(Nx*Nt*2*colors);
        int indx1, indx2;
        std::cout << "Printing P^T (transpose interpolator) " << std::endl;
        for(int b = 0; b < level.blocks_per_rank; b++){
            int bx = b / level.tblocks_per_rank;
		    int bt = b % level.tblocks_per_rank; 
            //Coordinates inside the block (bx,bt)
			int xini = level.x_elements*bx; int xfin = xini + level.x_elements;
			int tini = level.t_elements*bt; int tfin = tini + level.t_elements;
        for(int sc=0; sc<2;sc++){
        for(int cc = 0; cc<Ntest; cc++){
            std::cout << "(bx, bt, sc, cc) = (" << bx << ", " << bt << ", " << sc << ", " << cc << ")" << std::endl;
            indx1 = bx*level.tblocks_per_rank*Ntest*2 	+ bt*Ntest*2 + cc*2 + sc; 
            ev.val[indx1] = 1;
            level.P_vc(ev,column);
            ev.val[indx1] = 0;
        
		    for(int x=0; x<Nx; x++){
		    for(int t=0; t<Nt; t++){
		    for(int c=0; c<colors; c++){
		    for(int s=0; s<2; s++){
			    indx2 	= x*Nt*colors*2 			+ t*colors*2 + c*2 	+ s;
                std::cout << column.val[indx2] << " ";
            }
            }
            }
            }
            std::cout << std::endl;


        }
        }     
        }

    }


    //-------------------------------//
    if (mpi::rank2d == 0){
        spinor ev(Nx*Nt*2*colors); //Lives on the coarse lattice
        spinor column(level.blocks_per_rank*2*Ntest);
        int indx1, indx2;
        std::cout << "Printing P* (conjugate interpolator) " << std::endl;

        for(int x=0; x<Nx; x++){
		for(int t=0; t<Nt; t++){
		for(int c=0; c<colors; c++){
		for(int s=0; s<2; s++){
			indx2 	= x*Nt*colors*2 			+ t*colors*2 + c*2 	+ s;
            ev.val[indx2] = 1;
            level.Pdagg_v(ev,column);
            ev.val[indx2] = 0;

            for(int b = 0; b < level.blocks_per_rank; b++){
                int bx = b / level.tblocks_per_rank;
		        int bt = b % level.tblocks_per_rank; 
                for(int sc=0; sc<2;sc++){
                for(int cc = 0; cc<Ntest; cc++){
                    //std::cout << "(bx, bt, sc, cc) = (" << bx << ", " << bt << ", " << sc << ", " << cc << ")" << std::endl;
                    indx1 = bx*level.tblocks_per_rank*Ntest*2 	+ bt*Ntest*2 + cc*2 + sc; 
                    std::cout << column.val[indx1] << "   ";
                }
                }
            }
            std::cout << std::endl;
        }
        }
        }     
        }

    }


}

//P^+ P should be equal to the identity on the coarse level
//This only works if the test vectors are locally orthonormalized
void Check_PPdagg(const int& l, const spinor& U){
    Level level(l,U);
    int indxtv, indx;
    int Nt, Nx, colors, Ntest;
    Nt = level.Nt; Nx = level.Nx; colors = level.colors; Ntest = level.Ntest;
    static std::mt19937 randomInt(50); //Same seed for all the MPI copies
	std::uniform_real_distribution<double> distribution(-1.0, 1.0); //mu, standard deviation
    
    for(int cc = 0; cc < level.Ntest; cc++){
        for(int x=0; x<level.Nx; x++){
        for(int t=0; t<level.Nt; t++){
	    for(int c=0; c<level.colors; c++){
	    for(int s=0; s<2; s++){
            indx 	= x*Nt*colors*2 + t*colors*2 + c*2 	+ s;
			indxtv 	= indx*Ntest + cc;
            level.tvec.val[indxtv] = distribution(randomInt) + I_number * distribution(randomInt);;            
        }
    }
    }
    }      
    }

    spinor vc(level.blocks_per_rank*2*Ntest);
    spinor temp(Nx*Nt*2*colors);
    spinor PdaggPvc(level.blocks_per_rank*2*Ntest);

    for(int i = 0; i< level.blocks_per_rank*2*Ntest; i++)
        vc.val[i] = distribution(randomInt) + I_number * distribution(randomInt);
    




    level.P_vc(vc,temp);
    level.Pdagg_v(temp,PdaggPvc);

    for(int i = 0; i< level.blocks_per_rank*2*Ntest; i++){
        if (std::abs(vc.val[i]-PdaggPvc.val[i]) > 1e-8 ){
            std::cout << "P^+ P vc != vc" << std::endl;
            return; 
        } 
    }


}