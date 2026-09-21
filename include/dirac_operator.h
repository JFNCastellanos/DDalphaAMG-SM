#ifndef DIRAC_OPERATOR_INCLUDED
#define DIRAC_OPERATOR_INCLUDED
#include "boundary.h"
#include "halo_exchange.h"
#include "utils.h"



/*
	Dirac operator application D phi
	U: gauge configuration
	phi: spinor to apply the operator to
	m0: mass parameter
*/
void D_phi(const spinor& U, const spinor& phi, spinor& Dphi, const double& m0);


/*
	Dirac dagger operator application D^+ phi
	U: gauge configuration
	phi: spinor to apply the operator to
	m0: mass parameter
*/
void D_dagger_phi(const spinor& U, const spinor& phi, spinor& Dphi, const double& m0);


/*
	Application of D^+ D
	It calls the previous functions
*/
void D_D_dagger_phi(const spinor& U, const spinor& phi, spinor& Dphi, const double& m0);

namespace clover{
	//Computation of clover term
	//mu = 0 time direction, mu = 1 space direction
	inline void Compute_Q(const spinor& U){
		MPI_Status status;
		exchange_halo(U.val);
		//Corners we have to communicate manually 
		//Update top-right corner (needs bottom-left corner from diagonal rank)
		{
			int x0 = mpi::width_x, t0 = 1;
			int n0 = x0*(mpi::width_t+2)+t0;
			c_double bottom_left_s0 = U.val[2*n0];   //U_0(n-1+0)
			c_double bottom_left_s1 = U.val[2*n0+1]; //U_1(n-1+0)
			MPI_Send(&bottom_left_s0, 1, MPI_DOUBLE_COMPLEX, mpi::bot_left, 0, mpi::cart_comm);
			MPI_Recv(&bottom_left_s0, 1, MPI_DOUBLE_COMPLEX, mpi::top_right, 0, mpi::cart_comm, &status);

			MPI_Send(&bottom_left_s1, 1, MPI_DOUBLE_COMPLEX, mpi::bot_left, 1, mpi::cart_comm);
			MPI_Recv(&bottom_left_s1, 1, MPI_DOUBLE_COMPLEX, mpi::top_right, 1, mpi::cart_comm, &status);
			n0 = mpi::width_t+1; 
			U.val[2*n0] = bottom_left_s0;  
			U.val[2*n0+1] = bottom_left_s1;    
		}

		//Update bottom-left corner (needs top-right corner from diagonal rank)
		{
			int x0 = 1, t0 = mpi::width_t;
			int n0 = x0*(mpi::width_t+2)+t0;
			c_double top_right_s0 = U.val[2*n0];     //U_0(n+1-0)
			c_double top_right_s1 = U.val[2*n0+1];   //U_0(n+1-0)
			MPI_Send(&top_right_s0, 1, MPI_DOUBLE_COMPLEX, mpi::top_right, 2, mpi::cart_comm);
			MPI_Recv(&top_right_s0, 1, MPI_DOUBLE_COMPLEX, mpi::bot_left, 2, mpi::cart_comm, &status);

			MPI_Send(&top_right_s1, 1, MPI_DOUBLE_COMPLEX, mpi::top_right, 3, mpi::cart_comm);
			MPI_Recv(&top_right_s1, 1, MPI_DOUBLE_COMPLEX, mpi::bot_left, 3, mpi::cart_comm, &status);

			x0 = mpi::width_x+1; t0 = 0;
			n0 = x0*(mpi::width_t+2)+t0;
			U.val[2*n0] = top_right_s0;
			U.val[2*n0+1] = top_right_s1;
		}
		//Update top-left corner (needs bot-right corner from diagonal rank)
		{
			int x0 = mpi::width_x, t0 = mpi::width_t;
			int n0 = x0*(mpi::width_t+2)+t0;
			c_double bot_right_s0 = U.val[2*n0];   //U_0(n-1-0)
			c_double bot_right_s1 = U.val[2*n0+1]; //U_1(n-1-0)
			MPI_Send(&bot_right_s0, 1, MPI_DOUBLE_COMPLEX, mpi::bot_right, 4, mpi::cart_comm);
			MPI_Recv(&bot_right_s0, 1, MPI_DOUBLE_COMPLEX, mpi::top_left, 4, mpi::cart_comm, &status);

			MPI_Send(&bot_right_s1, 1, MPI_DOUBLE_COMPLEX, mpi::bot_right, 5, mpi::cart_comm);
			MPI_Recv(&bot_right_s1, 1, MPI_DOUBLE_COMPLEX, mpi::top_left, 5, mpi::cart_comm, &status);

			n0 = 0;
			U.val[2*n0]   = bot_right_s0;
			U.val[2*n0+1] = bot_right_s1;
		}


		int n, right, down, left, up;
		int x1_t_1, x_1_t_1, x_1_t1, x1_t1; //n-0+1, n-0-1, n+0-1, n+0+1
		c_double Umv, Uv_m, U_m_v, U_vm;  //U_{m,v}(n) + U_{v,-m}(n) + U_{-m,-v}(n) + U_{-v,m}(n)
		for(int x = 1; x<=mpi::width_x; x++){
			for(int t = 1; t<=mpi::width_t; t++){
				n = x*(mpi::width_t+2)+t;
				right = rpb[2*n]; down = rpb[2*n+1]; left = lpb[2*n]; up = lpb[2*n+1];
				int xp = x+1;
				int xm = x-1;
				int tp = t+1;
				int tm = t-1;
				x1_t_1 = xp*(mpi::width_t+2)+tm;//(x+1,t-1)
				x_1_t_1 = xm*(mpi::width_t+2)+tm; //(x-1,t-1)
				x_1_t1 = xm*(mpi::width_t+2)+tp; //(x-1,t+1)
				x1_t1 = xp*(mpi::width_t+2)+tp; //(x+1,t+1)
				


				//U_01(n) = U_0(n) U_1(n+0) U*_0(n+1) U*_1(n)
				Umv = U.val[2*n] * U.val[2*right+1] * std::conj(U.val[2*down]) * std::conj(U.val[2*n+1]);

				//U_{1-0}(n) = U_1(n) U*_0(n-0+1) U*_1(n-0) U_0(n-0)
				Uv_m = U.val[2*n+1] * std::conj(U.val[2*x1_t_1]) * std::conj(U.val[2*left+1]) * U.val[2*left];

				//U_{-0,-1}(n) = U*_0(n-0) U*_1(n-0-1) U_0(n-0-1) U_1(n-1)
				U_m_v = std::conj(U.val[2*left]) * std::conj(U.val[2*x_1_t_1+1]) * U.val[2*x_1_t_1] * U.val[2*up+1];

				//U_{-10}(n) = U*_1(n-1) U_0(n-1) U_1(n+0-1) U*_0(n)
				U_vm = std::conj(U.val[2*up+1]) * U.val[2*up] * U.val[2*x_1_t1+1] * std::conj(U.val[2*n]);

				clover::Q01[n] = Umv+Uv_m+U_m_v+U_vm;

			}
		}
	
	}

}



#endif