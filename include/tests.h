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
    level.orthonormalize();
    level.checkOrthogonality();

    spinor vc(level.blocks_per_rank*2*Ntest);
    spinor temp(Nx*Nt*2*colors);
    spinor PdaggPvc(level.blocks_per_rank*2*Ntest);

    for(int i = 0; i< level.blocks_per_rank*2*Ntest; i++)
        vc.val[i] = distribution(randomInt) + I_number * distribution(randomInt);
    

    level.P_vc(vc,temp);
    level.Pdagg_v(temp,PdaggPvc);

    for(int i = 0; i< level.blocks_per_rank*2*Ntest; i++){
        if (std::abs(vc.val[i]-PdaggPvc.val[i]) > 1e-8 ){
            if (mpi::rank2d == 0){
                std::cout << "P^+ P vc != vc" << std::endl;
                std::cout << "Either P is ill-defined or test vectors require orthonormalization" << std::endl;
            } 
            return; 
        } 
    }

    if (mpi::rank2d == 0)
        std::cout << "Test passed: P^+ P vc = vc" << std::endl;
    
}

//We check that the implementation of D_phi and D_operator have the exact same output on the fine level.
void test_Doperator_fine_level(const spinor& U){
    spinor phi(mpi::maxSizeH);
    spinor out1(mpi::maxSizeH);
    spinor out2(mpi::maxSizeH);
    for(int x = 1; x<=mpi::width_x; x++){
        for(int t = 1; t<=mpi::width_t; t++){
            int n = x*(mpi::width_t+2)+t;
            U.val[2*n]      = RandomU1();
            U.val[2*n+1]    = RandomU1();
            phi.val[2*n]      = RandomU1();
            phi.val[2*n+1]    = RandomU1();
        }
    }
    //exchange_halo(U.val);
    int l=0;
    Level level(l,U);
    level.orthonormalize();         //Orthonormalize test vectors
    level.checkOrthogonality();     //Checking orthogonality 


    D_phi(U,phi,out1,mass::m0);
    level.D_operator(phi,out2);
     for(int x = 1; x<=mpi::width_x; x++){
        for(int t = 1; t<=mpi::width_t; t++){
            int n = x*(mpi::width_t+2)+t;
            if (std::abs(out1.val[2*n]-out2.val[2*n]) > 1e-8 || std::abs(out1.val[2*n+1]-out2.val[2*n+1]) > 1e-8){
                 std::cout << "Both implementations of D don't coincide, rank " << mpi::rank2d << std::endl;
                if (mpi::rank2d == 0){
                    std::cout << "x " << x << " t " << t << std::endl;
                    std::cout << "D_phi      " << out1.val[2*n] << "    " <<  out1.val[2*n+1] << std::endl;
                    std::cout << "D_operator " << out2.val[2*n] << "    " <<  out2.val[2*n+1] << std::endl;
                    std::cout << std::endl;
                }
                return; 
            } 
            
        }
     }

    if (mpi::rank2d == 0)
        std::cout << "Both implementations of the Dirac operator yield the same result" << std::endl;


}


void rank_agglomeration_test(){

    if (mpi::size != 16){
         printf("This test is meant to be run with 16 processes.\n");
        MPI_Abort(mpi::cart_comm, EXIT_FAILURE);
    }

    int r = mpi::rank2d;
    int A[4] = {(r+1)*1,(r+1)*2,(r+1)*3,(r+1)*4};
    int rx = r / mpi::ranks_t;
    int rt = r % mpi::ranks_t;

   
    // Create the small group by including only processes 1 and 3 from the world group
    int const ranks_x_c = mpi::ranks_x/2;
    int const ranks_t_c = mpi::ranks_t/2;
    int const size_c   = ranks_x_c*ranks_t_c;
    int ranks_c[size_c];

    // Get the group or processes of the default communicator
    MPI_Group cart_comm_group;
    MPI_Comm_group(mpi::cart_comm, &cart_comm_group);
 

     if (0 <= rx && rx < ranks_x_c && 0 <= rt && rt < ranks_t_c)
        printf("A=[%d,%d,%d,%d] rank %d\n",A[0],A[1],A[2],A[3],r);


    int count=0;
    for(int rx=0; rx<ranks_x_c; rx++){
        for(int rt=0; rt<ranks_t_c; rt++){
            ranks_c[count++] = rx*mpi::ranks_t+rt; 
        }
    }

    MPI_Group coarse_group;
    MPI_Group_incl(cart_comm_group, size_c, ranks_c, &coarse_group);
 
    // Create the new communicator from that group of processes.
    MPI_Comm coarse_comm;
    MPI_Comm_create(mpi::cart_comm, coarse_group, &coarse_comm);
    
    if (coarse_comm != MPI_COMM_NULL){
        int coarse_rank;
        MPI_Comm_rank(coarse_comm, &coarse_rank);
        printf("Process %d in cart_comm is process %d in coarse_comm.\n", mpi::rank2d,coarse_rank);
        //Next step is to gather data from each rank in coarse_group to the root
        int buffer[4*size_c];
         std::cout << "Ranks in communicator " << size_c << std::endl;
        MPI_Gather(&A,4,MPI_INT,&buffer,4,MPI_INT,0,coarse_comm);
        if (coarse_rank == 0){     
            for(int i = 0; i<4*size_c; i++)
                std::cout << buffer[i] << ", ";
        }
        std::cout << std::endl;

    }

    
   
 

}