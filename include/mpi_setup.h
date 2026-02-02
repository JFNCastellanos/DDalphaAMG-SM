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

    //Diagonal ranks (needed for the staples)
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
    MPI_Type_vector(mpi::width_x, LV::dof, (mpi::width_t+2)*LV::dof, MPI_DOUBLE_COMPLEX, &column_type);
    MPI_Type_commit(&column_type);

}

inline void initializeMPI(){
    assignWidth();
    buildCartesianTopology();
    defineDataTypes();
}


#endif