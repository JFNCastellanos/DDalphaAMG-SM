#include "io.h"


void read_binary(const std::string& name,const spinor& U){
    using namespace LV;
    std::ifstream infile(name, std::ios::binary);
    if (!infile) {
       std::cerr << "File " << name << " not found " << std::endl;
        exit(1);
    }
    spinor GlobalConf((Nt+2)*(Nx+2)*2); //Temporary variable to store the full configuration

    MPI_Type_vector(mpi::width_x, mpi::width_t*2, (mpi::width_t+2)*2, MPI_DOUBLE_COMPLEX, &sub_block_type);
    MPI_Type_commit(&sub_block_type);

    MPI_Type_create_resized(sub_block_type, 0, sizeof(std::complex<double>), &sub_block_resized);
    MPI_Type_commit(&sub_block_resized);

    int counts[mpi::size];
    int displs[mpi::size];
    for(int r = 0; r < mpi::size; r++){
        counts[r] = 1;
        int rx = r / mpi::ranks_t; 
        int rt = r % mpi::ranks_t; 

        // Global starting position inside the buffer including halo (halo at index 0)
        int global_x_start = rx * Nx + 1; // +1 to skip halo
        int global_t_start = rt * Nt + 1; // +1 to skip halo
        // Displacement in complex-element units into buffer.val (including halo padding)
        displs[r] 	= (global_x_start * (mpi::width_t + 2) + global_t_start) * 2;
    }

    if (mpi::rank2d == 0){
        for (int x = 1; x <= Nx; x++) {
        for (int t = 1; t <= Nt; t++) {
            int n = x * (Nt+2) + t;
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
        //std::cout << "Binary conf read from " << name << std::endl;     
    }

    int input_ini_local = 2 * (mpi::width_t + 2 + 1);
    MPI_Scatterv(&GlobalConf.val[0], counts, displs, sub_block_resized,
            &U.val[input_ini_local], mpi::maxSize, MPI_DOUBLE_COMPLEX, 0, mpi::cart_comm);
}
