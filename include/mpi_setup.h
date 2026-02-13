#ifndef MPI_SETUP_H
#define MPI_SETUP_H
#include "variables.h"
#include "utils.h"

//Check that number of ranks on the x and t direction match the number of total ranks called.
inline void assignWidth(){
    if (mpi::ranks_t * mpi::ranks_x != mpi::size){
        if (mpi::rank == 0){
            std::cout << "ranks_t * ranks_x != total number of ranks" << std::endl;
            std::cout << mpi::ranks_t * mpi::ranks_x << " != " << mpi::size << std::endl;
        }
        exit(1);
    }
    //We do this to enforce an equal workload on each rank
    if (LV::Nx % mpi::ranks_x!= 0 ||LV::Nt % mpi::ranks_t != 0){
        if (mpi::rank == 0)
            std::cout << "Nx (Nt) is not exactly divisible by rank_x (rank_t)" << std::endl;
        exit(1);
    }
    mpi::width_x = LV::Nx/mpi::ranks_x;
    mpi::width_t = LV::Nt/mpi::ranks_t;
    mpi::maxSize = mpi::width_t * mpi::width_x;
    mpi::maxSizeH = LV::dof*(mpi::width_x+2)*(mpi::width_t+2); //With halos included
    mpi::sitesH = (mpi::width_x+2)*(mpi::width_t+2);
}
 
/*
 * Two-dimensional cartesian topology
 *                t                    2D parallelization
 *   0  +-------------------+  Nt   +-----------------------+
 *      |                   |       |                       |
 *      |                   |       | top-left top top-right|
 *      |                   |       |           |           |
 *   x  |                   |       |--left--rank2d--right--|
 *      |                   |       |           |           |
 *      |                   |       | bot-left bot bot-right|
 *      |                   |       |                       |
 *   Nx +-------------------+ Nt    +-----------------------+
 *                Nx
*/
inline void buildCartesianTopology(){
    int dims[2] = {mpi::ranks_x, mpi::ranks_t};
    int periods[2] = {1, 1}; // periodic in both dims
    int reorder = 1;         // allow rank reordering
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &mpi::cart_comm);

    //rank in the Cartesian communicator and its coordinates
    MPI_Comm_rank(mpi::cart_comm, &mpi::rank2d);
    MPI_Cart_coords(mpi::cart_comm, mpi::rank2d, 2, mpi::coords); // mpi::coords[0]=x coord, [1]=t coord
    
    //MPI_Cart_shift(cart_comm, Direction, Displacement, - direction,  +direction);
    //Along t direction
    MPI_Cart_shift(mpi::cart_comm, 1, 1, &mpi::left, &mpi::right);
    //Along x direction
    MPI_Cart_shift(mpi::cart_comm, 0, 1, &mpi::top , &mpi::bot);

    //Diagonal ranks (just in case)
    int coords_bot_left[2] = {mod(mpi::coords[0]+1,mpi::ranks_x), mod(mpi::coords[1]-1,mpi::ranks_t)}; //bot-left
    MPI_Cart_rank(mpi::cart_comm, coords_bot_left, &mpi::bot_left);

    int coords_bot_right[2] = {mod(mpi::coords[0]+1,mpi::ranks_x), mod(mpi::coords[1]+1,mpi::ranks_t)}; //bot-right
    MPI_Cart_rank(mpi::cart_comm, coords_bot_right, &mpi::bot_right);

    int coords_top_left[2] = {mod(mpi::coords[0]-1,mpi::ranks_x), mod(mpi::coords[1]-1,mpi::ranks_t)}; //top-left
    MPI_Cart_rank(mpi::cart_comm, coords_top_left, &mpi::top_left);

    int coords_top_right[2] = {mod(mpi::coords[0]-1,mpi::ranks_x), mod(mpi::coords[1]+1,mpi::ranks_t)}; //top-right
    MPI_Cart_rank(mpi::cart_comm, coords_top_right, &mpi::top_right);
    
    //printf("[MPI process %d] I am located at (%d, %d). Top %d bot %d right %d left %d bot-left %d bot-right %d top-left %d top-right %d  \n",
    //       mpi::rank2d, mpi::coords[0], mpi::coords[1], mpi::top, mpi::bot, mpi::right, mpi::left,
    //        mpi::bot_left,mpi::bot_right,mpi::top_left,mpi::top_right);
}




//Rank agglomeration
/*
 *                                   
 *                     t                                Rank coarsening
 *   0  +------------------------------+ Nt         +---------------------+        
 *      |  r0  |  r1   |  r2   |  r3   |            |          |          |
 *      |------|-------|-------|-------|            |  rank 0  |  rank 1  |   
 *   x  |  r4  |  r5   |  r6   |  r7   |            |          |          |
 *      |------|-------|-------|-------|            |----------|----------|         
 *      |  r8  |  r9   |  r10  |  r11  |            |          |          |
 *      |------|-------|-------|-------|            |  rank 2  |  rank 3  |
 *      |  r12 |  r13  |  r14  |  r15  |            |          |          |   
 *   Nx +------------------------------+ Nt         +---------------------+
 * 
 */
inline void coarseLevelCommunicators(){

    //Group of ranks that communicate on the finest level
    MPI_Comm_group(mpi::cart_comm, &mpi::cart_comm_group);

    //Number of agglomerated ranks
    mpi::ranks_x_c  = mpi::ranks_x/mpi::coarse_ranks_x;    // ranks_x/ranks on coarse grid on x
    mpi::ranks_t_c  = mpi::ranks_t/mpi::coarse_ranks_t;   
    mpi::size_c     = mpi::ranks_x_c*mpi::ranks_t_c;
    mpi::counts_coarse = new int[mpi::size_c];  
    mpi::displs_coarse = new int[mpi::size_c];
  
    int ranks_c[mpi::ranks_coarse_level][mpi::size_c];
    int rcl_x, rcl_t;
    int rx_ini, rx_fin, rt_ini, rt_fin;

    for(int rcl=0; rcl<mpi::ranks_coarse_level; rcl++){
        rcl_x = rcl / mpi::ranks_t_c;
        rcl_t = rcl % mpi::ranks_t_c;
        
        rx_ini = rcl_x*mpi::ranks_x_c; rx_fin = rx_ini + mpi::ranks_x_c; 
        rt_ini = rcl_t*mpi::ranks_t_c; rt_fin = rt_ini + mpi::ranks_t_c; 
        int count = 0;
        for(int rx=rx_ini; rx<rx_fin; rx++){
            for(int rt=rt_ini; rt<rt_fin; rt++){
                ranks_c[rcl][count++] = rx*mpi::ranks_t+rt; 
                mpi::rank_dictionary[rx*mpi::ranks_t+rt] = rcl;
            }
        }        
    }

    //Create groups and communicators for the rank agglomeration 
    for(int rcl=0; rcl<mpi::ranks_coarse_level; rcl++){
        MPI_Group_incl(mpi::cart_comm_group, mpi::size_c, ranks_c[rcl], &mpi::coarse_group[rcl]);
        MPI_Comm_create(mpi::cart_comm, mpi::coarse_group[rcl], &mpi::coarse_comm[rcl]);
    }

    mpi::Nx_coarse_rank = mpi::width_x*mpi::ranks_x_c;
    mpi::Nt_coarse_rank = mpi::width_t*mpi::ranks_t_c;
  
    // Prepare counts and displacements (displacements in complex-element units)
    for (int r = 0; r < mpi::size_c; r++) {
        mpi::counts_coarse[r] = 1; // one instance of recv_domain_resized per contributing rank
        int rx = r / mpi::ranks_t_c; // coarse-group x coordinate
        int rt = r % mpi::ranks_t_c; // coarse-group t coordinate
        // Global starting position inside the buffer including halo (halo at index 0)
        int global_x_start = rx * mpi::width_x + 1; // +1 to skip halo
        int global_t_start = rt * mpi::width_t + 1; // +1 to skip halo
        // Displacement in complex-element units into buffer.val (including halo padding)
        mpi::displs_coarse[r] = (global_x_start * (mpi::Nt_coarse_rank + 2) + global_t_start) * 2;
    }

    //Now we prepare a communicator to send/receive data among the fine-grid ranks that will be working on the coarse levels
    //i.e. the root_ranks of the agglomerated communicators.
    int coarse_level_ranks[mpi::ranks_coarse_level];
    int coarse_rank, commID;
    int count = 0;

    MPI_Comm temp_comm; //Temporary communicator
    MPI_Group temp_group;

    
    //We store the fine-grid ranks that will be working on the coarse levels
    for(int r=0; r<mpi::size; r++){	   
        int rx = r / mpi::ranks_t; // coarse-group x coordinate
        int rt = r % mpi::ranks_t; // coarse-group t coordinate
        int rc = (rx%mpi::ranks_x_c) * mpi::ranks_t_c + (rt%mpi::ranks_t_c);
        if (rc == 0){
            coarse_level_ranks[count++] = r;  
        }
    }

    
    MPI_Group_incl(mpi::cart_comm_group, mpi::ranks_coarse_level, coarse_level_ranks, &temp_group);
    MPI_Comm_create(mpi::cart_comm, temp_group, &temp_comm);
    

    
    int dims[2] = {mpi::coarse_ranks_x, mpi::coarse_ranks_t};
    int periods[2] = {1, 1}; // periodic in both dims
    int reorder = 1;         // allow rank reordering

    //temp_comm is only defined for its member ranks
    if (temp_comm != MPI_COMM_NULL){
        MPI_Comm_rank(temp_comm, &mpi::coarse_rank2d);
        MPI_Cart_create(temp_comm, 2, dims, periods, reorder, &mpi::comm_coarse_level);
        //MPI_Cart_shift(cart_comm, Direction, Displacement, - direction,  +direction);
        //Along t direction
        MPI_Cart_shift(mpi::comm_coarse_level, 1, 1, &mpi::left_c, &mpi::right_c);
        //Along x direction
        MPI_Cart_shift(mpi::comm_coarse_level, 0, 1, &mpi::top_c , &mpi::bot_c);

    }

    /*
    if (mpi::comm_coarse_level != MPI_COMM_NULL){
        printf("[MPI process on cart_comm %d, on comm_coarse_level %d. Neighbors top_c %d bot_c %d right_c %d left_c %d\n",
        mpi::rank2d,mpi::coarse_rank2d, mpi::top_c, mpi::bot_c, mpi::right_c, mpi::left_c);
    }
    */
    
}


inline void defineDataTypes(){
    //Create a new data type for the blocks corresponding to each rank
    /*  
    *              width_t
    *          ---------------     
    *          |             |
    *          |             |
    * width_x  |             |
    *          |             |
    *          |             |
    *          ---------------
    */
    //int MPI_Type_vector(int block_count, int block_length, int stride, MPI_Datatype old_datatype, MPI_Datatype* new_datatype);
    MPI_Type_vector(mpi::width_x, mpi::width_t, LV::Nt, MPI_DOUBLE_COMPLEX, &sub_block_type);
    MPI_Type_commit(&sub_block_type);

    //Resize the data type to use scatterV properly
    int extent = mpi::width_t;
    MPI_Type_create_resized(sub_block_type, 0, extent * sizeof(std::complex<double>), &sub_block_resized);
    MPI_Type_commit(&sub_block_resized);

    //Create datatype for the halo exchange
    MPI_Type_vector(mpi::width_x, LV::dof, (mpi::width_t+2)*LV::dof, MPI_DOUBLE_COMPLEX, &mpi::column_type[0]);
    MPI_Type_commit(&mpi::column_type[0]);

    //Data type for the elements of a spinor inside a rank. 
    //The datatype does not include the halo, but assumes the spinor to be sent has it
    MPI_Type_vector(mpi::width_x,mpi::width_t*2,2*(mpi::width_t+2),MPI_DOUBLE_COMPLEX, &local_domain);
    MPI_Type_commit(&local_domain);

    //The displacement of local_domain_resized is in units of std::complex<double>
    MPI_Type_create_resized(local_domain, 0, sizeof(std::complex<double>), &local_domain_resized);
    MPI_Type_commit(&local_domain_resized);

    // Gather inner domains from all ranks in the coarse communicator
    // Buffer has size (Nx_coarse_rank+2)*(Nt_coarse_rank+2)*2
    // Create a type that matches the global buffer layout (strided by full global row including halo)
    MPI_Type_vector(mpi::width_x,                   // number of rows to place per rank
                    2 * mpi::width_t,               // elements per row (complex numbers)
                    2 * (mpi::Nt_coarse_rank + 2),  // stride between rows in global buffer (complex elements) including halo
                    MPI_DOUBLE_COMPLEX,
                    &coarse_domain);
    MPI_Type_commit(&coarse_domain);

    // Resize type so displacements are specified in units of one complex element
    MPI_Type_create_resized(coarse_domain, 0, sizeof(std::complex<double>), &coarse_domain_resized);
    MPI_Type_commit(&coarse_domain_resized);


}

inline void initializeMPI(){
    assignWidth();
    buildCartesianTopology();
    
    if (mpi::ranks_x > 2 && mpi::ranks_t >2) 
        coarseLevelCommunicators();

    defineDataTypes();
}

inline void defineColumnType(const int& l, const int& Nx, const int& Nt, const int& DOF){
    //Create datatype for the halo exchange at level l
    MPI_Type_vector(Nx, DOF, (Nt+2)*DOF, MPI_DOUBLE_COMPLEX, &mpi::column_type[l]);
    MPI_Type_commit(&mpi::column_type[l]);
}


#endif