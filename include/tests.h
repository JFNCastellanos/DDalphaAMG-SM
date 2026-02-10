#include "dirac_operator.h"
#include "level.h"
#include "sap.h"
#include "gather_scatter.h"
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
    const int &blocks_x = level.xblocks_per_rank;
    const int &blocks_t = level.tblocks_per_rank;
    Nt = level.Nt; Nx = level.Nx; colors = level.colors; Ntest = level.Ntest;
    
    int count = 0;
    for(int cc = 0; cc < level.Ntest; cc++){
        for(int x=1; x<=level.Nx; x++){
        for(int t=1; t<=level.Nt; t++){
	    for(int c=0; c<level.colors; c++){
	    for(int s=0; s<2; s++){
            indx 	= x*(Nt+2)*colors*2 + t*colors*2 + c*2 	+ s;
			indxtv 	= indx*Ntest + cc;
            level.tvec[cc].val[indx] = indxtv+1;
           /* if (mpi::rank2d == 0){
                std::cout << "(x,t,c,s) = " << x << ", " << t << ", " << c << ", " << s <<  std::endl;
                std::cout << "indx " << indxtv << std::endl;
            }*/
            
            count++;
        }
    }
    }
    }      
    }

    if (mpi::rank2d == 0){
        std::cout << "Rank " << mpi::rank2d << std::endl;
        std::cout << "Nx " << Nx << " Nt " << Nt << " colors " << colors << " Ntest " << Ntest << std::endl;
        std::cout << "Blocks per rank " << level.blocks_per_rank << std::endl;
        std::cout << "X elements inside block " << level.x_elements << " T elements " << level.t_elements << std::endl;
        std::cout << "count " << count << std::endl;
    
    }

    
    if (mpi::rank2d == 0){
        spinor ev((blocks_t+2)*(blocks_x+2)*2*Ntest); //Lives on the coarse lattice
        spinor column((Nx+2)*(Nt+2)*2*colors);
        int indx1, indx2;
        std::cout << "Printing P^T (transpose interpolator) " << std::endl;
        for(int b = 0; b < level.blocks_per_rank; b++){
            int bx = b / blocks_t; //This are the indices for the blocks, not for the coarse spinor
		    int bt = b % blocks_t; 
            int bx_shifted = bx+1; //The shifted versions are used for the coarse spinor with halos
            int bt_shifted = bt+1;
            //Coordinates inside the block (bx,bt) for a spinor with halos
			int xini = level.x_elements*bx+1; int xfin = xini + level.x_elements;
			int tini = level.t_elements*bt+1; int tfin = tini + level.t_elements;
        for(int sc=0; sc<2;sc++){
        for(int cc = 0; cc<Ntest; cc++){
            std::cout << "(bx, bt, sc, cc) = (" << bx << ", " << bt << ", " << sc << ", " << cc << ")" << std::endl;
            indx1 = bx_shifted*(blocks_t+2)*Ntest*2 	+ bt_shifted*Ntest*2 + cc*2 + sc; 
            ev.val[indx1] = 1;
            level.P_vc(ev,column);
            ev.val[indx1] = 0;
        
		    for(int x=1; x<=Nx; x++){
		    for(int t=1; t<=Nt; t++){
		    for(int c=0; c<colors; c++){
		    for(int s=0; s<2; s++){
			    indx2 	= x*(Nt+2)*colors*2 			+ t*colors*2 + c*2 	+ s;
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
        spinor ev((Nx+2)*(Nt+2)*2*colors); //Lives on the coarse lattice
        spinor column((blocks_t+2)*(blocks_x+2)*2*Ntest);
        int indx1, indx2;
        std::cout << "Printing P* (conjugate interpolator) " << std::endl;

        for(int x=1; x<=Nx; x++){
		for(int t=1; t<=Nt; t++){
		for(int c=0; c<colors; c++){
		for(int s=0; s<2; s++){
			indx2 	= x*(Nt+2)*colors*2 			+ t*colors*2 + c*2 	+ s;
            ev.val[indx2] = 1;
            level.Pdagg_v(ev,column);
            ev.val[indx2] = 0;

            for(int b = 0; b < level.blocks_per_rank; b++){
                int bx = b / blocks_t;
		        int bt = b % blocks_t; 
                int bx_shifted = bx+1;
                int bt_shifted = bt+1;
                for(int sc=0; sc<2;sc++){
                for(int cc = 0; cc<Ntest; cc++){
                    //std::cout << "(bx, bt, sc, cc) = (" << bx << ", " << bt << ", " << sc << ", " << cc << ")" << std::endl;
                    indx1 = bx_shifted*(blocks_t+2)*Ntest*2 	+ bt_shifted*Ntest*2 + cc*2 + sc; 
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
        for(int x=1; x<=level.Nx; x++){
        for(int t=1; t<=level.Nt; t++){
	    for(int c=0; c<level.colors; c++){
	    for(int s=0; s<2; s++){
            indx 	= x*(Nt+2)*colors*2 + t*colors*2 + c*2 	+ s;
            level.tvec[cc].val[indx] = distribution(randomInt) + I_number * distribution(randomInt);            
        }
    }
    }
    }      
    }
    level.orthonormalize();
    level.checkOrthogonality();

    spinor vc(level.blocks_per_rank*2*Ntest);
    spinor temp((Nx+2)*(Nt+2)*2*colors);
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
            U.val[2*n]        = RandomU1();
            U.val[2*n+1]      = RandomU1();
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

//Testing the rank agglomeration function.
void rank_agglomeration_test(){

    if (mpi::size != 16){
         printf("This test is meant to be run with 16 processes.\n");
        MPI_Abort(mpi::cart_comm, EXIT_FAILURE);
    }
    int r = mpi::rank2d;
    int A[4] = {(r+1)*1,(r+1)*2,(r+1)*3,(r+1)*4};

    for(int i = 0; i < mpi::size; i++) {
        MPI_Barrier(mpi::cart_comm);
        if (i == mpi::rank2d) 
            printf("A=[%d,%d,%d,%d] rank %d\n",A[0],A[1],A[2],A[3],r);    
    }

    MPI_Barrier(mpi::cart_comm);

    
    for(int cg=0; cg<mpi::ranks_coarse_level; cg++){
        if (mpi::coarse_comm[cg] != MPI_COMM_NULL){
            int coarse_rank;
            MPI_Comm_rank(mpi::coarse_comm[cg], &coarse_rank);
            printf("Process %d in cart_comm is process %d in coarse_comm %d.\n", mpi::rank2d,coarse_rank,cg);
           
            int buffer[4*mpi::size_c];
            MPI_Gather(&A,4,MPI_INT,&buffer,4,MPI_INT,0,mpi::coarse_comm[cg]);
            if (coarse_rank == 0){  
                std::cout << "Coarse group " << cg << std::endl;   
                for(int i = 0; i<4*mpi::size_c; i++)
                    std::cout << buffer[i] << ", ";
                }
            std::cout << std::endl;
        }
    }

}


//Test inner_domain and recv_domain datatypes for gathering data.
void gather_vector_test(){
    if (mpi::size != 16){
         printf("This test is meant to be run with 16 processes.\n");
        MPI_Abort(mpi::cart_comm, EXIT_FAILURE);
    }
    spinor input((mpi::width_t+2)*(mpi::width_x+2)*2);
    int n;
    for(int x = 1; x<=mpi::width_x; x++){
        for(int t = 1; t<=mpi::width_t; t++){
            n = x*(mpi::width_t+2)+t;
            for(int mu=0; mu<2; mu++){
                input.val[2*n+mu] = 2*n+mu + mpi::rank2d;
            }        
        }
    }


    for(int i = 0; i < mpi::size; i++) {
        MPI_Barrier(mpi::cart_comm);
        if (i == mpi::rank2d) {
            std::cout << "rank " << mpi::rank2d << std::endl;
            for(int x = 1; x<=mpi::width_x; x++){
                for(int t = 1; t<=mpi::width_t; t++){
                    int n = x*(mpi::width_t+2) + t;
                    std::cout << "[" << input.val[2*n] << ", " << input.val[2*n+1] << "], ";
                }
                std::cout << std::endl;
            } 
        }
    }

  
    spinor buffer((mpi::Nx_coarse_rank+2)*(mpi::Nt_coarse_rank+2)*2);

    gather_to_coarse_rank(input, buffer);

   
    int local_rank;
    int commID = mpi::rank_dictionary[mpi::rank2d];  
    MPI_Comm_rank(mpi::coarse_comm[commID], &local_rank);
    for(int i = 0; i <mpi::size; i++) {
        MPI_Barrier(mpi::cart_comm);
        if (i == mpi::rank2d && local_rank == 0) {
            std::cout << "\nGather in rank " << mpi::rank2d << std::endl;
            for(int x = 0; x<mpi::Nx_coarse_rank+2; x++){
                for(int t = 0; t<mpi::Nt_coarse_rank+2; t++){
                    int n = x*(mpi::Nt_coarse_rank+2) + t;
                    std::cout << "[" << buffer.val[2*n] << ", " << buffer.val[2*n+1] << "], ";
                }
                std::cout << std::endl;
            } 
        }
    }
        
}

void scatter_vector_test(){
    if (mpi::size != 16){
         printf("This test is meant to be run with 16 processes.\n");
        MPI_Abort(mpi::cart_comm, EXIT_FAILURE);
    }
    spinor input((mpi::Nx_coarse_rank+2)*(mpi::Nt_coarse_rank+2)*2);
    spinor buffer((mpi::width_t+2)*(mpi::width_x+2)*2);

	int commID = mpi::rank_dictionary[mpi::rank2d];
    int local_rank;
    MPI_Comm_rank(mpi::coarse_comm[commID], &local_rank);

    int n;
    for(int x = 1; x<=mpi::Nx_coarse_rank; x++){
        for(int t = 1; t<=mpi::Nt_coarse_rank; t++){
            n = x*(mpi::Nt_coarse_rank+2)+t;
            for(int mu=0; mu<2; mu++){
                input.val[2*n+mu] = 2*n+mu + mpi::rank2d;
            }        
        }
    }


    for(int i = 0; i < mpi::size; i++) {
        MPI_Barrier(mpi::cart_comm);
        if (i == mpi::rank2d && local_rank == 0) {
            std::cout << "rank " << mpi::rank2d << std::endl;
            for(int x = 1; x<=mpi::Nx_coarse_rank; x++){
                for(int t = 1; t<=mpi::Nt_coarse_rank; t++){
                    int n = x*(mpi::Nt_coarse_rank+2) + t;
                    std::cout << "[" << input.val[2*n] << ", " << input.val[2*n+1] << "], ";
                }
                std::cout << std::endl;
            } 
        }
    }

    scatter_to_local_rank_from_coarse_rank(input, buffer);
    
    for(int i = 0; i <mpi::size; i++) {
        MPI_Barrier(mpi::cart_comm);
        if (i == mpi::rank2d) {
            std::cout << "\nGather in rank " << mpi::rank2d << std::endl;
            for(int x = 0; x<mpi::width_x+2; x++){
                for(int t = 0; t<mpi::width_t+2; t++){
                    int n = x*(mpi::width_t+2) + t;
                    std::cout << "[" << buffer.val[2*n] << ", " << buffer.val[2*n+1] << "], ";
                }
                std::cout << std::endl;
            } 
        }
    }
        
}


