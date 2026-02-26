#include "io.h"


void read_binary(const std::string& name,const spinor& U){
    using namespace LV;
    std::ifstream infile(name, std::ios::binary);
    if (!infile) {
       std::cerr << "File " << name << " not found " << std::endl;
        exit(1);
    }
    spinor GlobalConf((Nt+2)*(Nx+2)*2); //Temporary variable to store the full configuration


    int counts[mpi::size];
    int displs[mpi::size];
    for(int r = 0; r < mpi::size; r++){
        counts[r] = 1;
        int rx = r / mpi::ranks_t; 
        int rt = r % mpi::ranks_t; 

        // Global starting position inside the buffer including halo (halo at index 0)
        int global_x_start = rx * mpi::width_x + 1; // +1 to skip halo
        int global_t_start = rt * mpi::width_t + 1; // +1 to skip halo
        // Displacement in complex-element units into buffer.val (including halo padding)
        displs[r] = (global_x_start * (LV::Nt + 2) + global_t_start) * 2;

        //if (mpi::rank2d == 0)
        //    std::cout << "displ " << displs[r] << std::endl;
    }

    if (mpi::rank2d == 0){
        for (int x = 1; x <= LV::Nx; x++) {
        for (int t = 1; t <= LV::Nt; t++) {
            int n = x * (LV::Nt+2) + t;
            for (int mu = 0; mu < 2; mu++) {
                int x_read, t_read, mu_read;
                double re, im;
                infile.read(reinterpret_cast<char*>(&x_read), sizeof(int));
                infile.read(reinterpret_cast<char*>(&t_read), sizeof(int));
                infile.read(reinterpret_cast<char*>(&mu_read), sizeof(int));
                infile.read(reinterpret_cast<char*>(&re), sizeof(double));
                infile.read(reinterpret_cast<char*>(&im), sizeof(double));
                GlobalConf.val[2*n+mu] = c_double(re, im); 
            }
        }
        }
        infile.close();
        std::cout << "Binary conf read from " << name << std::endl;     
    }

    int input_ini_local = 2 * (mpi::width_t + 2 + 1);
    MPI_Scatterv(&GlobalConf.val[0], counts, displs, global_conf_resized, &U.val[input_ini_local],1,
        	local_conf_resized, 0, mpi::cart_comm);

}


void broadcast_file_name(std::string& File){
    //Broadcast file name from rank 0 to other ranks
    int filename_len = 0;
    if (mpi::rank == 0) 
        filename_len = static_cast<int>(File.size());

    MPI_Bcast(&filename_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mpi::rank != 0) 
        File.resize(filename_len);
    
    MPI_Bcast(File.data(), filename_len, MPI_CHAR, 0, MPI_COMM_WORLD);
}