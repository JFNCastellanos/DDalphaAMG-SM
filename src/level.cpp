#include "level.h"

//Prolongator times a vector on the coarse grid
void Level::P_vc(const spinor& vc,spinor& out){
	for(int i = 0; i < (Nx+2)*(Nt+2)*colors*2; i++)
		out.val[i] = 0.0; 

	//Botch vc and out include their corresponding halo 
	int bx, bt, bx_shifted, bt_shifted;
	//(bx,bt) are the block coordinates without halo
	//(bx_shifted ,bt_shifted) consider the halo
	int xini, tini, xfin, tfin;
	int idxout, idxv; //Vectorized index of out, v.
	
	//In case we have a lattice block crossing the MPI ranks of the fine level 
	if (ranks_per_block > 1){
		//Define vc across the ranks -> Gather tv and out to root rank of coarse comm -> apply the P_vc operation
		//->return out to the original 2d communicator.

		//Spinor on the coarse grid.
		//Vc will only be called from with this size  spinor vc_coarse_rank((xblocks_per_coarse_rank+2)*(tblocks_per_coarse_rank+2)*Ntest*2);	
		std::vector<spinor> gathered_tvec(Ntest,spinor((Nt_coarse_rank+2)*(Nx_coarse_rank+2)*2*colors));	//Gather test vectors data from other ranks
		//Gather test vectors from all the ranks living inside the coarse rank.
		spinor gathered_out((Nt_coarse_rank+2)*(Nx_coarse_rank+2)*2*colors);
		for(int cc=0;cc<Ntest;cc++)
			gather_to_coarse_rank(tvec[cc],gathered_tvec[cc]);

		for(int b = 0; b<blocks_per_coarse_rank; b++){
			bx = b / tblocks_per_coarse_rank;
			bt = b % tblocks_per_coarse_rank;
			bx_shifted = bx+1;
			bt_shifted = bt+1;
			//Coordinates inside the block (bx,bt) for a spinor with halo
			xini = x_elements*bx+1; xfin = xini + x_elements;
			tini = t_elements*bt+1; tfin = tini + t_elements;
			for(int cc = 0; cc < Ntest; cc++){
				for(int x=xini; x<xfin; x++){
				for(int t=tini; t<tfin; t++){	
				for(int c=0; c<colors; c++){
				for(int s=0; s<2; s++){
					idxout 	= x*(Nt_coarse_rank+2)*colors*2 					+ t*colors*2 		 + c*2 	+ s;
					idxv 	= bx_shifted*(tblocks_per_coarse_rank+2)*Ntest*2 	+ bt_shifted*Ntest*2 + cc*2 + s;
					gathered_out.val[idxout] += gathered_tvec[cc].val[idxout] * vc.val[idxv]; 
				}
				}
				}
				}
			}
		}
		scatter_to_local_rank_from_coarse_rank(gathered_out,out);
	}
	else{
		for(int b = 0; b<blocks_per_rank; b++){
			bx = b / tblocks_per_rank;
			bt = b % tblocks_per_rank;	
			bx_shifted = bx+1;
			bt_shifted = bt+1;
			//Coordinates inside the block (bx,bt) for a spinor with halo
			xini = x_elements*bx+1; xfin = xini + x_elements;
			tini = t_elements*bt+1; tfin = tini + t_elements;
			for(int cc = 0; cc < Ntest; cc++){
				for(int x=xini; x<xfin; x++){
				for(int t=tini; t<tfin; t++){	
				for(int c=0; c<colors; c++){
				for(int s=0; s<2; s++){
					idxout 	= x*(Nt+2)*colors*2 						+ t*colors*2 + c*2 	+ s;
					idxv 	= bx_shifted*(tblocks_per_rank+2)*Ntest*2 	+ bt_shifted*Ntest*2 + cc*2 + s;
					out.val[idxout] += tvec[cc].val[idxout] * vc.val[idxv]; 
				}
				}
				}
				}
			}
		}
	}
	
}
	


//Restriction operator times a spinor on the fine grid
void Level::Pdagg_v(const spinor& v,spinor& out) {
	//out lives on the coarse grid
	int bx, bt, bx_shifted, bt_shifted;
	int xini, tini, xfin, tfin;
	int idxout, idxv; //Vectorized index of out, v.
	//In case we have a lattice block crossing the MPI ranks of the fine level 
	if (ranks_per_block > 1){	
		for(int i = 0; i < (xblocks_per_coarse_rank+2)*(tblocks_per_coarse_rank+2)*2*Ntest; i++)
			out.val[i]= 0.0; //Initialize the output spinor
		
		std::vector<spinor> gathered_tvec(Ntest,spinor((Nt_coarse_rank+2)*(Nx_coarse_rank+2)*2*colors));	//Gather test vectors data from other ranks
		spinor gathered_v((Nt_coarse_rank+2)*(Nx_coarse_rank+2)*2*colors);
		for(int cc=0;cc<Ntest;cc++)
			gather_to_coarse_rank(tvec[cc],gathered_tvec[cc]);
		gather_to_coarse_rank(v,gathered_v);

		for (int b = 0; b<blocks_per_coarse_rank; b++) {	
			bx = b / tblocks_per_coarse_rank;
			bt = b % tblocks_per_coarse_rank; 
			bx_shifted = bx+1;
			bt_shifted = bt+1;
			xini = x_elements*bx+1; xfin = xini + x_elements;
			tini = t_elements*bt+1; tfin = tini + t_elements;
			for(int cc=0; cc<Ntest; cc++){
				for(int x=xini; x<xfin; x++){
				for(int t=tini; t<tfin; t++){	
				for(int c=0; c<colors; c++){
				for(int s=0; s<2; s++){
					idxout 	= bx_shifted*(tblocks_per_coarse_rank+2)*Ntest*2 	+ bt_shifted*Ntest*2 + cc*2 + s;
					idxv 	= x*(Nt_coarse_rank+2)*colors*2 					+ t*colors*2 + c*2 	+ s;
					out.val[idxout] += std::conj(gathered_tvec[cc].val[idxv]) * gathered_v.val[idxv];
				}	
				}
				}
				}
			}
		}
		//Out already lives on the coarse rank
	}
	else{
		for(int i = 0; i < (xblocks_per_rank+2)*(tblocks_per_rank+2)*2*Ntest; i++)
			out.val[i]= 0.0; //Initialize the output spinor
		for (int b = 0; b<blocks_per_rank; b++) {	
			bx = b / tblocks_per_rank;
			bt = b % tblocks_per_rank; 
			bx_shifted = bx+1;
			bt_shifted = bt+1;
			xini = x_elements*bx+1; xfin = xini + x_elements;
			tini = t_elements*bt+1; tfin = tini + t_elements;
			for(int cc=0; cc<Ntest; cc++){
				for(int x=xini; x<xfin; x++){
				for(int t=tini; t<tfin; t++){	
				for(int c=0; c<colors; c++){
				for(int s=0; s<2; s++){
					idxout 	= bx_shifted*(tblocks_per_rank+2)*Ntest*2 		+ bt_shifted*Ntest*2 + cc*2 + s;
					idxv 	= x*(Nt+2)*colors*2 							+ t*colors*2 + c*2 	+ s;
					out.val[idxout] += std::conj(tvec[cc].val[idxv]) * v.val[idxv];
				}	
				}
				}
				}
			}
		}
	}
}

//Local orthonormalization.
void Level::orthonormalize(){	
    //Given a set of test vectors, returns a local orthonormalization.
    //For the set of test vectors (v1^(1)|v2^(1)|...|v_Nv^(1)|...|v1^(Nagg)|...|v_Nv^(Nagg)), we
    //orthonormalize the sets {v1^{a}, ..., vn^(a)} for each aggregate.
    //This follows the steps from Section 3.1 of A. Frommer et al "Adaptive Aggregation-Based Domain Decomposition 
    //Multigrid for the Lattice Wilson-Dirac Operator", SIAM, 36 (2014).

	//Orthonormalization by applying Gram-Schmidt
	c_double proj; 
	c_double norm;

	int bx, bt, bx_shifted, bt_shifted, xini, xfin, tini, tfin;
	int indx;
	//Lattice blocks
	for (int b = 0; b<blocks_per_rank; b++) {	
		bx = b / tblocks_per_rank;
		bt = b % tblocks_per_rank; 
		//----Coordinates of the elements inside block----//
		xini = x_elements*bx+1; xfin = xini + x_elements;
		tini = t_elements*bt+1; tfin = tini + t_elements; 
		//-----------------------------------------------//
		//Spin defining the aggregate for a particular block
		for (int s = 0; s < 2; s++) {
			//Looping over the test vectors
			for (int nt = 0; nt < Ntest; nt++) {
				for (int ntt = 0; ntt < nt; ntt++) {
					proj = 0;
					for(int x=xini; x<xfin; x++){
					for(int t=tini; t<tfin; t++){	
					for(int c=0; c<colors; c++){
						indx = x*(Nt+2)*colors*2 + t*colors*2 + c*2 + s;
						proj += tvec[nt].val[indx] * std::conj(tvec[ntt].val[indx]);

					}
					}
					}
					for(int x=xini; x<xfin; x++){
					for(int t=tini; t<tfin; t++){	
					for(int c=0; c<colors; c++){
						indx = x*(Nt+2)*colors*2 + t*colors*2 + c*2 + s;
						tvec[nt].val[indx] -= proj * tvec[ntt].val[indx];

					}
					}
					}
				}
				//normalize
				norm = 0.0;
				for(int x=xini; x<xfin; x++){
				for(int t=tini; t<tfin; t++){	
				for(int c=0; c<colors; c++){
					indx = x*(Nt+2)*colors*2 + t*colors*2 + c*2 + s;
					norm += tvec[nt].val[indx] * std::conj(tvec[nt].val[indx]);

				}
				}
				}
				norm = sqrt(std::real(norm)) + 0.0*I_number;
				for(int x=xini; x<xfin; x++){
				for(int t=tini; t<tfin; t++){	
				for(int c=0; c<colors; c++){
					indx = x*(Nt+2)*colors*2 + t*colors*2 + c*2 + s;
					tvec[nt].val[indx] /= norm;

				}
				}
				} 
				
			}
		} 	
	}
}

void Level::makeDirac(){
	c_double P[2][2][2], M[2][2][2]; 
	//P = 1 + sigma
	P[0][0][0] = 1.0; P[0][0][1] = 1.0;
	P[0][1][0] = 1.0; P[0][1][1] = 1.0; 

	P[1][0][0] = 1.0; P[1][0][1] = -I_number;
	P[1][1][0] = I_number; P[1][1][1] = 1.0; 

	//M = 1- sigma
	M[0][0][0] = 1.0; M[0][0][1] = -1.0;
	M[0][1][0] = -1.0; M[0][1][1] = 1.0; 

	M[1][0][0] = 1.0; M[1][0][1] = I_number;
	M[1][1][0] = -I_number; M[1][1][1] = 1.0; 

	int n;
	exchange_halo(U.val);

	for(int x = 1; x<=Nx; x++){
	for(int t = 1; t<=Nt; t++){
		n = x*(Nt+2)+t;
	for(int alf=0; alf<2;alf++){
	for(int bet=0; bet<2;bet++){
	for(int c = 0; c<colors; c++){
	for(int b = 0; b<colors; b++){
		G1.val[getG1index(n,alf,bet,c,b)] 	  = 0; //This coefficient is not used at level 0
		G2.val[getG2G3index(n,alf,bet,c,b,0)] = 0; G2.val[getG2G3index(n,alf,bet,c,b,1)] = 0;
		G3.val[getG2G3index(n,alf,bet,c,b,0)] = 0; G3.val[getG2G3index(n,alf,bet,c,b,1)] = 0;
		//For level = 0 
		for(int mu : {0,1}){
			G2.val[getG2G3index(n,alf,bet,c,b,mu)] = 0.5 * M[mu][alf][bet] * U.val[2*n+mu];
			G3.val[getG2G3index(n,alf,bet,c,b,mu)] = 0.5 * P[mu][alf][bet] * std::conj(U.val[2*lpb[2*n+mu]+mu]);
		}
		
	}
	}
	}
	}

	}
	}
	
		
}


//Exchange halo for spinor v among the working ranks at the current level
void Level::exchange_halo_l(const spinor& v){
	using namespace mpi;
	//The current implementation only works when the aggregates don't cross the ranks
    int row_size = DOF * Nt;
	int send_start, recv_start;   

    //Send top row to top rank. Receive top row from bot rank.
	//(x*(Nt+2)+t)*DOF +dof
	send_start = ((Nt+2) + 1)*DOF;   //x=1, t=1
	recv_start = ((Nx+1)*(Nt+2)+1)*DOF;	//x=Nx+1, t=1
    MPI_Sendrecv(&v.val[send_start], row_size, MPI_DOUBLE_COMPLEX, top, 0,
        &v.val[recv_start], row_size, MPI_DOUBLE_COMPLEX, bot, 0,
        cart_comm, MPI_STATUS_IGNORE);

    //Send bot row to bot rank. Receive bot row from top rank.
	send_start = (Nx*(Nt+2)+1)*DOF;	//x=Nx, t=1
	recv_start = 1*DOF;				//x=0,	t=1
    MPI_Sendrecv(&v.val[send_start], row_size, MPI_DOUBLE_COMPLEX, bot, 1,
        &v.val[recv_start], row_size, MPI_DOUBLE_COMPLEX, top, 1,
        cart_comm, MPI_STATUS_IGNORE);

    //Send left column to left rank. Receive left column from right rank. 
	send_start = ((Nt+2) + 1)*DOF;		//x=1,	t=1
	recv_start = ((Nt+2)+(Nt+1))*DOF;	//x=1,	t=Nt+1
    MPI_Sendrecv(&v.val[send_start], 1, column_type[level], left, 2,
    &v.val[recv_start], 1, column_type[level], right, 2,
    cart_comm, MPI_STATUS_IGNORE);

    //Send right column to right rank. Receive right column from left rank. 
	send_start = ((Nt+2)+Nt)*DOF;		//x=1,	t=Nt
	recv_start = (Nt+2)*DOF;			//x=1,	t=0
    MPI_Sendrecv(&v.val[send_start], 1, column_type[level], right, 3,
    &v.val[recv_start], 1, column_type[level], left, 3,
    cart_comm, MPI_STATUS_IGNORE);
}


//Dirac operator at the current level
void Level::D_operator(const spinor& v, spinor& out){	

	exchange_halo_l(v); //Communicate halos

	int indx, indx1, indx2, n;
	//n only runs in the interior of the lattice domain

	for(int x = 1; x<=Nx; x++){
	for(int t = 1; t<=Nt; t++){
		n = x*(Nt+2)+t;
		for(int alf = 0; alf<2; alf++){
		for(int c = 0; c<colors; c++){
			indx = n*colors*2+c*2+alf;
			out.val[indx] = (mass::m0+2)*v.val[indx];
		for(int bet = 0; bet<2; bet++){
		for(int b = 0; b<colors; b++){
			indx1 = n*colors*2+b*2+bet;
			out.val[indx] -= G1.val[getG1index(n,alf,bet,c,b)] * v.val[indx1];
			for(int mu:{0,1}){
				indx1 = rpb_l(x,t,mu)*colors*2+b*2+bet;
				indx2 = lpb_l(x,t,mu)*colors*2+b*2+bet;
				out.val[indx] -= ( 	G2.val[getG2G3index(n,alf,bet,c,b,mu)] * rsign_l(t,mu) * v.val[indx1]
								+ 	G3.val[getG2G3index(n,alf,bet,c,b,mu)] * lsign_l(t,mu) * v.val[indx2] 
								);
			}
		}
		}
		}
		}
	}
	}
	
}




/*
	Make coarse gauge links. They will be used in the next level as G1, G2 and G3.
*/
void Level::makeCoarseLinks(Level& next_level){
	//Make gauge links for level l
	std::vector<spinor> &w = tvec;
	for(int cc=0; cc<Ntest;cc++)
		exchange_halo_l(w[cc]);	//We exchange halos for the test vectors.
	


	c_double wG2, wG3;
	spinor &A_coeff = next_level.G1; 
	spinor &B_coeff = next_level.G2;
	spinor &C_coeff = next_level.G3;
	int indxA; int indxBC[2]; //Indices for A, B and C coefficients
	int block_r;
	int block_l;
	//p and s are the colors at the coarse level
	//c and b are the colors at the current level
	int bx, bt, bx_shifted, bt_shifted, block;
	int xini, xfin, tini, tfin;
	int n; 
	int indx;
	int rn, ln; //right and left neighbors
	int rb, lb; //right and left blocks
	for(int b = 0; b<blocks_per_rank; b++){
		bx = b / tblocks_per_rank;
		bt = b % tblocks_per_rank;	
		bx_shifted = bx+1;
		bt_shifted = bt+1;
		block = (bx_shifted)*(tblocks_per_rank+2) + bt_shifted;//Indexing for a coarse spinors with halo
		//Coordinates inside the block (bx,bt) for a spinor with halo
		xini = x_elements*bx+1; xfin = xini + x_elements;
		tini = t_elements*bt+1; tfin = tini + t_elements;
	for(int alf=0; alf<2; alf++){
	for(int bet=0; bet<2; bet++){
	for(int p = 0; p<Ntest; p++){
	for(int s = 0; s<Ntest; s++){
		indxA 		= getAindex(block,alf,bet,p,s); 	//Indices for the next level
		indxBC[0] 	= getBCindex(block,alf,bet,p,s,0);
		indxBC[1] 	= getBCindex(block,alf,bet,p,s,1);
		A_coeff.val[indxA] = 0;
		B_coeff.val[indxBC[0]] = 0; B_coeff.val[indxBC[1]] = 0;
		C_coeff.val[indxBC[0]] = 0; C_coeff.val[indxBC[1]] = 0;
		//Elements inside the lattice blocks
		for(int x=xini;x<xfin;x++){
		for(int t=tini;t<tfin;t++){
			n = x*(Nt+2)+t;
			for(int c = 0; c<colors; c++){
			for(int b = 0; b<colors; b++){
				//[w*_p^(block,alf)]_{c,alf}(x) [A(x)]^{alf,bet}_{c,b} [w_s^{block,bet}]_{b,bet}(x)
				indx = n*colors*2 + c*2 + alf;
				A_coeff.val[indxA] += std::conj(w[p].val[indx]) * G1.val[getG1index(n,alf,bet,c,b)] * w[s].val[n*colors*2 + b*2 + bet];
				for(int mu : {0,1}){
					rn = rpb_l(x,t,mu); //(x,t)+hat{mu}
					ln = lpb_l(x,t,mu); //(x,t)-hat{mu}
					rb = next_level.rpb_l(bx_shifted,bt_shifted,mu); //(bx,bt)+hat{mu}
					lb = next_level.lpb_l(bx_shifted,bt_shifted,mu); //(bx,bt)-hat{mu}
					getLatticeBlock(rn, block_r); //block_r: block where rn lives
					getLatticeBlock(ln, block_l); //block_l: block where ln lives

					wG2 = std::conj(w[p].val[indx]) * G2.val[getG2G3index(n,alf,bet,c,b,mu)]; 
					wG3 = std::conj(w[p].val[indx]) * G3.val[getG2G3index(n,alf,bet,c,b,mu)];

					//Only diff from zero when n+hat{mu} in Block(x)
					if (block_r == block){
						A_coeff.val[indxA] 			+= wG2 * w[s].val[rn*colors*2 + b*2 + bet];
					}
					//Only diff from zero when n+hat{mu} in Block(x+hat{mu})
					else if (block_r == rb){
						B_coeff.val[indxBC[mu]] 	+= wG2 * w[s].val[rn*colors*2 + b*2 + bet]; 
					}
					//Only diff from zero when n-hat{mu} in Block(x)
					if (block_l == block){
						A_coeff.val[indxA] 			+= wG3 * w[s].val[ln*colors*2 + b*2 + bet];
					}
					//Only diff from zero when n-hat{mu} in Block(x-hat{mu})
					else if (block_l == lb){
						C_coeff.val[indxBC[mu]] 	+= wG3 * w[s].val[ln*colors*2 + b*2 + bet];
					}
				}//mu loop
			}//b loop
			}//c loop
		}//x inside block
		}//t inside block	
	} //s
	} //p
	} //bet
	} //alf
	} //block
}
//Local Dc used for SAP
/*
void Level::SAP_level_l::D_local(const spinor& in, spinor& out, const int& block){

	int RightPB_0, blockRPB_0; //Right periodic boundary in the 0-direction
    int RightPB_1, blockRPB_1; //Right periodic boundary in the 1-direction
    int LeftPB_0, blockLPB_0; //Left periodic boundary in the 0-direction
    int LeftPB_1, blockLPB_1; //Left periodic boundary in the 1-direction

	int RightPB, blockRPB;
	int LeftPB, blockLPB;

	spinor phi_RPB = spinor(2,c_vector(spins*colors,0));
	spinor phi_LPB = spinor(2,c_vector(spins*colors,0));

	//x is the index inside of the Schwarz block	
	for(int x = 0; x<lattice_sites_per_block; x++){
		int n = Blocks[block][x]; //n is the index of the lattice point in the original lattice

		//If the neighbor sites are part of the same block we consider them for D 
		//otherwise, we ignore them, i.e. we fix them to zero.
		for(int mu: {0,1}){
			getMandBlock(RightPB_l[parent->level][n][mu], RightPB, blockRPB);
			getMandBlock(LeftPB_l[parent->level][n][mu], LeftPB, blockLPB);
			if(blockRPB == block){
				for(int dof=0; dof<parent->DOF; dof++)
					phi_RPB[mu][dof] = in[RightPB][dof]; 
			}
        	else {
				for(int dof=0; dof<parent->DOF; dof++)
					phi_RPB[mu][dof] = 0;
			}

			if(blockLPB == block){
				for(int dof=0; dof<parent->DOF; dof++)
					phi_LPB[mu][dof] = in[LeftPB][dof]; 
			}
        	else {
				for(int dof=0; dof<parent->DOF; dof++)
					phi_LPB[mu][dof] = 0;
			}

		}

		//Local application of the Dirac operator		
		for(int alf = 0; alf<2; alf++){
		for(int c = 0; c<colors; c++){
				out[x][2*c+alf] = (mass::m0+2)*in[x][2*c+alf];
				FLOPS += da + dcm;
			for(int bet = 0; bet<2; bet++){
			for(int b = 0; b<colors; b++){
				out[x][2*c+alf] -= parent->G1[parent->getG1index(n,alf,bet,c,b)] * in[x][2*b+bet];
				FLOPS += ca+cm;
				for(int mu:{0,1}){
					out[x][2*c+alf] -= 
						( parent->G2[parent->getG2G3index(n,alf,bet,c,b,mu)] * SignR_l[parent->level][n][mu] * phi_RPB[mu][2*b+bet]
						+ parent->G3[parent->getG2G3index(n,alf,bet,c,b,mu)] * SignL_l[parent->level][n][mu] * phi_LPB[mu][2*b+bet]
						);
					FLOPS += ca+ca+4*cm;
				}

			}
			}
		}
		}
		
	}


}
*/



void Level::checkOrthogonality() {
	int bx, bt, xini, xfin, tini, tfin;
	c_double dot_product;
	int indx;
	for (int b = 0; b<blocks_per_rank; b++) {	
		bx = b / tblocks_per_rank;
		bt = b % tblocks_per_rank; 
		//----Coordinates of the elements inside block----//
		xini = x_elements*bx+1; xfin = xini + x_elements;
		tini = t_elements*bt+1; tfin = tini + t_elements; 
		//-----------------------------------------------//
		//Spin defining the aggregate for a particular block
		for (int s = 0; s < 2; s++) {
			//Looping over the test vectors
			for (int nt = 0; nt < Ntest; nt++) {
			for (int ntt = 0; ntt < Ntest; ntt++) {
				dot_product = 0.0;
				for(int x=xini; x<xfin; x++){
				for(int t=tini; t<tfin; t++){	
				for(int c=0; c<colors; c++){
					indx = x*(Nt+2)*colors*2 + t*colors*2 + c*2 + s;
					dot_product += std::conj(tvec[nt].val[indx]) * tvec[ntt].val[indx]; //v_nt . v_ntt
				}
				}
				}
				if (std::abs(dot_product) > 1e-8 && nt!=ntt) {
					if (mpi::rank2d == 0){
						std::cout << "Block " << b << " spin " << s << std::endl;
						std::cout << "Level " << level << std::endl;
						std::cout << "Test vectors " << nt << " and " << ntt << " are not orthogonal: " << dot_product << std::endl;
					}
					exit(1);
				}
				else if(std::abs(dot_product-1.0) > 1e-8 && nt==ntt){
					if (mpi::rank2d == 0){
						std::cout << "Level " << level << std::endl;
						std::cout << "Test vector " << nt << " not normalized " << dot_product << std::endl;
					}
					exit(1);
				}
			}
			}
		}
	}
	
	if (mpi::rank2d == 0)
		std::cout << "Test vectors on level " << level << " are orthonormalized " << std::endl;
}



//Local orthonormalization.
void Level::orthonormalize_v2(){	
    //Given a set of test vectors, returns a local orthonormalization.
    //For the set of test vectors (v1^(1)|v2^(1)|...|v_Nv^(1)|...|v1^(Nagg)|...|v_Nv^(Nagg)), we
    //orthonormalize the sets {v1^{a}, ..., vn^(a)} for each aggregate.
    //This follows the steps from Section 3.1 of A. Frommer et al "Adaptive Aggregation-Based Domain Decomposition 
    //Multigrid for the Lattice Wilson-Dirac Operator", SIAM, 36 (2014).

	//Orthonormalization by applying Gram-Schmidt
	c_double proj; 
	c_double norm;

	int bx, bt, bx_shifted, bt_shifted, xini, xfin, tini, tfin;
	int indx;
	//Lattice blocks
	std::vector<spinor> gathered_tvec(Ntest,spinor((Nt_coarse_rank+2)*(Nx_coarse_rank+2)*2*colors));	//Gather test vectors data from other ranks
	for(int cc=0;cc<Ntest;cc++)
		gather_to_coarse_rank(tvec[cc],gathered_tvec[cc]);

	int commID = mpi::rank_dictionary[mpi::rank2d];
	int coarse_rank;
	MPI_Comm_rank(mpi::coarse_comm[commID], &coarse_rank);

	if (coarse_rank == 0){
		for (int b = 0; b<blocks_per_coarse_rank; b++) {	
			bx = b / tblocks_per_coarse_rank;
			bt = b % tblocks_per_coarse_rank; 
			//----Coordinates of the elements inside block----//
			xini = x_elements*bx+1; xfin = xini + x_elements;
			tini = t_elements*bt+1; tfin = tini + t_elements; 
			//-----------------------------------------------//
			//Spin defining the aggregate for a particular block
			for (int s = 0; s < 2; s++) {
				//Looping over the test vectors
				for (int nt = 0; nt < Ntest; nt++) {
					for (int ntt = 0; ntt < nt; ntt++) {
						proj = 0;
						for(int x=xini; x<xfin; x++){
						for(int t=tini; t<tfin; t++){	
						for(int c=0; c<colors; c++){
							indx = x*(Nt_coarse_rank+2)*colors*2 + t*colors*2 + c*2 + s;
							proj += gathered_tvec[nt].val[indx] * std::conj(gathered_tvec[ntt].val[indx]);

						}
						}
						}
						for(int x=xini; x<xfin; x++){
						for(int t=tini; t<tfin; t++){	
						for(int c=0; c<colors; c++){
							indx = x*(Nt_coarse_rank+2)*colors*2 + t*colors*2 + c*2 + s;
							gathered_tvec[nt].val[indx] -= proj * gathered_tvec[ntt].val[indx];

						}
						}
						}
					}
					//normalize
					norm = 0.0;
					for(int x=xini; x<xfin; x++){
					for(int t=tini; t<tfin; t++){	
					for(int c=0; c<colors; c++){
						indx = x*(Nt_coarse_rank+2)*colors*2 + t*colors*2 + c*2 + s;
						norm += gathered_tvec[nt].val[indx] * std::conj(gathered_tvec[nt].val[indx]);

					}
					}
					}
					norm = sqrt(std::real(norm)) + 0.0*I_number;
					for(int x=xini; x<xfin; x++){
					for(int t=tini; t<tfin; t++){	
					for(int c=0; c<colors; c++){
						indx = x*(Nt_coarse_rank+2)*colors*2 + t*colors*2 + c*2 + s;
						gathered_tvec[nt].val[indx] /= norm;

					}
					}
					} 
				
				}
			} 	
		}
	}
		
	for(int cc=0;cc<Ntest;cc++)
		scatter_to_local_rank_from_coarse_rank(gathered_tvec[cc],tvec[cc]);
}

void Level::checkOrthogonality_v2(){
	int bx, bt, xini, xfin, tini, tfin;
	c_double dot_product;
	int indx;
	//Lattice blocks
	std::vector<spinor> gathered_tvec(Ntest,spinor((Nt_coarse_rank+2)*(Nx_coarse_rank+2)*2*colors));	//Gather test vectors data from other ranks
	for(int cc=0;cc<Ntest;cc++)
		gather_to_coarse_rank(tvec[cc],gathered_tvec[cc]);

	int commID = mpi::rank_dictionary[mpi::rank2d];
	int coarse_rank;
	MPI_Comm_rank(mpi::coarse_comm[commID], &coarse_rank);
	//For other ranks the gathered_tvec remain as zero
	if (coarse_rank == 0){
		for (int b = 0; b<blocks_per_coarse_rank; b++) {	
			bx = b / tblocks_per_coarse_rank;
			bt = b % tblocks_per_coarse_rank; 
			//----Coordinates of the elements inside block----//
			xini = x_elements*bx+1; xfin = xini + x_elements;
			tini = t_elements*bt+1; tfin = tini + t_elements; 
			//-----------------------------------------------//
			//Spin defining the aggregate for a particular block
			for (int s = 0; s < 2; s++) {
				//Looping over the test vectors
				for (int nt = 0; nt < Ntest; nt++) {
				for (int ntt = 0; ntt < Ntest; ntt++) {
					dot_product = 0.0;
					for(int x=xini; x<xfin; x++){
					for(int t=tini; t<tfin; t++){	
					for(int c=0; c<colors; c++){
						indx = x*(Nt_coarse_rank+2)*colors*2 + t*colors*2 + c*2 + s;
						dot_product += std::conj(gathered_tvec[nt].val[indx]) * gathered_tvec[ntt].val[indx]; //v_nt . v_ntt
					}
					}
					}

					if (std::abs(dot_product) > 1e-8 && nt!=ntt) {
						if (mpi::rank2d == 0){
							std::cout << "Block " << b << " spin " << s << std::endl;
							std::cout << "Level " << level << std::endl;
							std::cout << "Test vectors " << nt << " and " << ntt << " are not orthogonal: " << dot_product << std::endl;
						}
					//exit(1);
					}
					else if(std::abs(dot_product-1.0) > 1e-8 && nt==ntt){
						if (mpi::rank2d == 0){
							std::cout << "Level " << level << std::endl;
							std::cout << "Test vector " << nt << " not normalized " << dot_product << std::endl;
						}
					//exit(1);
					}
				}
				}
			}
		}
	}
	
	if (coarse_rank == 0)
		std::cout << "Test vectors on level " << level << " are orthonormalized " << " from comm " << commID << std::endl;
}