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
			int xini = level.x_elements*bx+1; int xfin = xini + level.x_elements;
			int tini = level.t_elements*bt+1; int tfin = tini + level.t_elements;
        for(int sc=0; sc<2;sc++){
        for(int cc = 0; cc<Ntest; cc++){
            std::cout << "(bx, bt, sc, cc) = (" << bx << ", " << bt << ", " << sc << ", " << cc << ")" << std::endl;
            indx1 = bx*level.tblocks_per_rank*Ntest*2 	+ bt*Ntest*2 + cc*2 + sc; 
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
        spinor ev(Nx*Nt*2*colors); //Lives on the coarse lattice
        spinor column(level.blocks_per_rank*2*Ntest);
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
        for(int x=1; x<=level.Nx; x++){
        for(int t=1; t<=level.Nt; t++){
	    for(int c=0; c<level.colors; c++){
	    for(int s=0; s<2; s++){
            indx 	= x*(Nt+2)*colors*2 + t*colors*2 + c*2 	+ s;
            level.tvec[cc].val[indx] = distribution(randomInt) + I_number * distribution(randomInt);;            
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


void gather_vector_test(){
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

    int Nx_tot_sites = mpi::width_x*mpi::ranks_x_c;
    int Nt_tot_sites = mpi::width_t*mpi::ranks_t_c;
    spinor buffer((Nx_tot_sites+2)*(Nt_tot_sites+2)*2);

    int root_rank = 0;  //Root rank inside the communicator agglomerating ranks
	int commID = mpi::rank_dictionary[mpi::rank2d];
    int counts_recv[mpi::size_c];
    int displs[mpi::size_c];

    /*
    MPI_Type_vector(int block_count,
        int block_length,
        int stride,
        MPI_Datatype old_datatype,
        MPI_Datatype* new_datatype);
    */

    MPI_Type_vector(mpi::width_x,
        mpi::width_t*2,
        2*(mpi::width_t+2),
        MPI_DOUBLE_COMPLEX,
        &inner_domain);
    MPI_Type_commit(&inner_domain);

    int input_ini = 2 * (mpi::width_t + 2 + 1);  // start of [1,1] in local input



    // Gather inner domains from all ranks in the coarse communicator
    // Now include halos in the global receive buffer: buffer has size (Nx_tot_sites+2)*(Nt_tot_sites+2)*2
    // Create a recv type that matches the global buffer layout (strided by full global row including halo)
    MPI_Datatype recv_domain;
    MPI_Type_vector(mpi::width_x,            // number of rows to place per rank
                    2 * mpi::width_t,        // elements per row (complex numbers)
                    2 * (Nt_tot_sites + 2),  // stride between rows in global buffer (complex elements) including halo
                    MPI_DOUBLE_COMPLEX,
                    &recv_domain);
    MPI_Type_commit(&recv_domain);

    // Resize recv type so displacements are specified in units of one complex element
    MPI_Datatype recv_domain_resized;
    MPI_Type_create_resized(recv_domain, 0, sizeof(std::complex<double>), &recv_domain_resized);
    MPI_Type_commit(&recv_domain_resized);

    // Prepare counts and displacements (displacements in complex-element units)
    int input_ini_local = 2 * (mpi::width_t + 2 + 1); // start of [1,1] in local input (complex elements)
    for (int r = 0; r < mpi::size_c; r++) {
        counts_recv[r] = 1; // one instance of recv_domain_resized per contributing rank

        int rx = r / mpi::ranks_t_c; // coarse-group x coordinate
        int rt = r % mpi::ranks_t_c; // coarse-group t coordinate

        // Global starting position inside the buffer including halo (halo at index 0)
        int global_x_start = rx * mpi::width_x + 1; // +1 to skip halo
        int global_t_start = rt * mpi::width_t + 1; // +1 to skip halo

        // Displacement in complex-element units into buffer.val (including halo padding)
        displs[r] = (global_x_start * (Nt_tot_sites + 2) + global_t_start) * 2;
    }

    // Use Gatherv: send 1 instance of the local strided type, receive into the resized global type
    MPI_Gatherv(&input.val[input_ini_local],
                1,
                inner_domain,
                &buffer.val[0],
                counts_recv,
                displs,
                recv_domain_resized,
                root_rank,
                mpi::coarse_comm[commID]);

    MPI_Type_free(&recv_domain);
    MPI_Type_free(&recv_domain_resized);

    
    for(int i = 0; i <mpi::size; i++) {
        MPI_Barrier(mpi::cart_comm);
        if (i == mpi::rank2d) {
            std::cout << "\nGather in rank " << mpi::rank2d << std::endl;
            for(int x = 0; x<Nx_tot_sites+2; x++){
                for(int t = 0; t<Nt_tot_sites+2; t++){
                    int n = x*(Nt_tot_sites+2) + t;
                    std::cout << "[" << buffer.val[2*n] << ", " << buffer.val[2*n+1] << "], ";
                }
                std::cout << std::endl;
            } 
        }
    }
        
}