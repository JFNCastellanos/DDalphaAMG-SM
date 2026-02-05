#include "level.h"

//Prolongator times a vector on the coarse grid
void Level::P_vc(const spinor& vc,spinor& out){
	//Loop over columns
	for(int i = 0; i < Nx*Nt*colors*2; i++)
		out.val[i] = 0.0; //Initialize the output spinor

	int bx, bt;
	int xini, tini, xfin, tfin;
	int idxout, idxv, idxtv; //Vectorized index of out, v and test vector
	
	for(int b = 0; b<blocks_per_rank; b++){
		bx = b / tblocks_per_rank;
		bt = b % tblocks_per_rank;	
		//Coordinates inside the block (bx,bt)
		xini = x_elements*bx; xfin = xini + x_elements;
		tini = t_elements*bt; tfin = tini + t_elements;
		for(int cc = 0; cc < Ntest; cc++){
			for(int x=xini; x<xfin; x++){
			for(int t=tini; t<tfin; t++){	
			for(int c=0; c<colors; c++){
			for(int s=0; s<2; s++){
				idxout 	= x*Nt*colors*2 				+ t*colors*2 + c*2 	+ s;
				idxtv 	= idxout*Ntest+cc;	 
				idxv 	= bx*tblocks_per_rank*Ntest*2 	+ bt*Ntest*2 + cc*2 + s;
				out.val[idxout] += tvec.val[idxtv] * vc.val[idxv];
			}
			}
			}
			}
		}
	}
}
	


//Restriction operator times a spinor on the fine grid
void Level::Pdagg_v(const spinor& v,spinor& out) {
	for(int i = 0; i < blocks_per_rank*2*Ntest; i++)
		out.val[i]= 0.0; //Initialize the output spinor

	int bx, bt;
	int xini, tini, xfin, tfin;
	int idxout, idxv, idxtv; //Vectorized index of out, v and test vector

	for (int b = 0; b<blocks_per_rank; b++) {	
		bx = b / tblocks_per_rank;
		bt = b % tblocks_per_rank; 
		xini = x_elements*bx; xfin = xini + x_elements;
		tini = t_elements*bt; tfin = tini + t_elements;
		for(int cc=0; cc<Ntest; cc++){
			for(int x=xini; x<xfin; x++){
			for(int t=tini; t<tfin; t++){	
			for(int c=0; c<colors; c++){
			for(int s=0; s<2; s++){
				idxout 	= bx*tblocks_per_rank*Ntest*2 		+ bt*Ntest*2 + cc*2 + s;
				idxv 	= x*Nt*colors*2 					+ t*colors*2 + c*2 	+ s;
				idxtv 	= idxv*Ntest 						+ cc;
				out.val[idxout] += std::conj(tvec.val[idxtv]) * v.val[idxv];
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

	int bx, bt, xini, xfin, tini, tfin;
	int indx;
	//Lattice blocks
	for (int b = 0; b<blocks_per_rank; b++) {	
		bx = b / tblocks_per_rank;
		bt = b % tblocks_per_rank; 
		//----Coordinates of the elements inside block----//
		xini = x_elements*bx; xfin = xini + x_elements;
		tini = t_elements*bt; tfin = tini + t_elements; 
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
						indx = x*Nt*colors*2 + t*colors*2 + c*2 + s;
						proj += tvec.val[indx*Ntest+nt] * std::conj(tvec.val[indx*Ntest+ntt]);

					}
					}
					}
					for(int x=xini; x<xfin; x++){
					for(int t=tini; t<tfin; t++){	
					for(int c=0; c<colors; c++){
						indx = x*Nt*colors*2 + t*colors*2 + c*2 + s;
						tvec.val[indx*Ntest+nt] -= proj * tvec.val[indx*Ntest+ntt];

					}
					}
					}
				}
				//normalize
				norm = 0.0;
				for(int x=xini; x<xfin; x++){
				for(int t=tini; t<tfin; t++){	
				for(int c=0; c<colors; c++){
					indx = x*Nt*colors*2 + t*colors*2 + c*2 + s;
					norm += tvec.val[indx*Ntest+nt] * std::conj(tvec.val[indx*Ntest+nt]);

				}
				}
				}
				norm = sqrt(std::real(norm)) + 0.0*I_number;
				for(int x=xini; x<xfin; x++){
				for(int t=tini; t<tfin; t++){	
				for(int c=0; c<colors; c++){
					indx = x*Nt*colors*2 + t*colors*2 + c*2 + s;
					tvec.val[indx*Ntest+nt] /= norm;

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

	int nini = (Nt+2)+1; int nfin= Nx*(Nt+2)+Nt;
	for(int n=nini; n<=nfin; n++){
	for(int alf=0; alf<2;alf++){
	for(int bet=0; bet<2;bet++){
	for(int c = 0; c<colors; c++){
	for(int b = 0; b<colors; b++){
		G1.val[getG1index(n,alf,bet,c,b)] = 0;//This coefficient is not used at level 0
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



//Dirac operator at the current level
void Level::D_operator(const spinor& v, spinor& out){	

	exchange_halo(v.val); //Communicate halos

	int indx, indx1, indx2;
	//for(int x = 1; x<=width_x; x++)
	//	for(int t = 1; t<=width_t; t++)
			//n = x*(width_t+2)+t;

	//n only runs in the interior of the lattice domain
	int nini = (Nt+2)+1; int nfin= Nx*(Nt+2)+Nt;

	for(int n = nini; n<=nfin;n++){
	for(int alf = 0; alf<2; alf++){
	for(int c = 0; c<colors; c++){
		indx = n*colors*2+c*2+alf;
		out.val[indx] = (mass::m0+2)*v.val[indx];
	for(int bet = 0; bet<2; bet++){
	for(int b = 0; b<colors; b++){
		indx1 = n*colors*2+b*2+bet;
		out.val[indx] -= G1.val[getG1index(n,alf,bet,c,b)] * v.val[indx1];
		for(int mu:{0,1}){
			indx1 = rpb[2*n+mu]*colors*2+b*2+bet;
			indx2 = lpb[2*n+mu]*colors*2+b*2+bet;
			out.val[indx] -= ( 	G2.val[getG2G3index(n,alf,bet,c,b,mu)] * rsign[2*n+mu] * v.val[indx1]
							+ 		G3.val[getG2G3index(n,alf,bet,c,b,mu)] * lsign[2*n+mu] * v.val[indx2] 
							);
		}
	}
	}
	}
	}
	}

	/*
	for(int x = 0; x<Ntot;x++){
	for(int alf = 0; alf<2; alf++){
	for(int c = 0; c<colors; c++){
		out[x][2*c+alf] = (mass::m0+2)*v[x][2*c+alf];
	for(int bet = 0; bet<2; bet++){
	for(int b = 0; b<colors; b++){
		out[x][2*c+alf] -= G1[getG1index(x,alf,bet,c,b)] * v[x][2*b+bet];
		for(int mu:{0,1}){
			out[x][2*c+alf] -= ( G2[getG2G3index(x,alf,bet,c,b,mu)] * SignR_l[level][2*x+mu] * v[RightPB_l[level][2*x+mu]][2*b+bet]
							+ G3[getG2G3index(x,alf,bet,c,b,mu)] * SignL_l[level][2*x+mu] * v[LeftPB_l[level][2*x+mu]][2*b+bet] );
		}
	}
	}
	}
	}
	}
	*/

}




/*
	Make coarse gauge links. They will be used in the next level as G1, G2 and G3.
*/
/*
void Level::makeCoarseLinks(Level& next_level){
	//Make gauge links for level l
	std::vector<spinor> &w = interpolator_columns;
	c_double wG2, wG3;
	c_vector &A_coeff = next_level.G1; 
	c_vector &B_coeff = next_level.G2;
	c_vector &C_coeff = next_level.G3;
	int indxA; int indxBC[2]; //Indices for A, B and C coefficients
	int block_r;
	int block_l;
	//p and s are the coarse colors
	//c and b are the colors at the current level
	for(int x=0; x<NBlocks; x++){
	for(int alf=0; alf<2;alf++){
	for(int bet=0; bet<2;bet++){
	for(int p = 0; p<Ntest; p++){
	for(int s = 0; s<Ntest; s++){
		indxA = getAindex(x,alf,bet,p,s); //Indices for the next level
		indxBC[0] = getBCindex(x,alf,bet,p,s,0);
		indxBC[1] = getBCindex(x,alf,bet,p,s,1);
		A_coeff[indxA] = 0;
		B_coeff[indxBC[0]] = 0; B_coeff[indxBC[1]] = 0;
		C_coeff[indxBC[0]] = 0; C_coeff[indxBC[1]] = 0;
		for(int n : LatticeBlocks[x]){
			for(int c = 0; c<colors; c++){
			for(int b = 0; b<colors; b++){
			
				//[w*_p^(block,alf)]_{c,alf}(x) [A(x)]^{alf,bet}_{c,b} [w_s^{block,bet}]_{b,bet}(x)
			A_coeff[indxA] += std::conj(w[p][n][2*c+alf]) * G1[getG1index(n,alf,bet,c,b)] * w[s][n][2*b+bet];
			for(int mu : {0,1}){
				getLatticeBlock(RightPB_l[level][n][mu], block_r); //block_r: block where RightPB_l[n][mu] lives
				getLatticeBlock(LeftPB_l[level][n][mu], block_l); //block_l: block where LeftPB_l[n][mu] lives
				wG2 = std::conj(w[p][n][2*c+alf]) * G2[getG2G3index(n,alf,bet,c,b,mu)]; 
				wG3 = std::conj(w[p][n][2*c+alf]) * G3[getG2G3index(n,alf,bet,c,b,mu)];
				FLOPS += cm*2;
				
				//Only diff from zero when n+hat{mu} in Block(x)
				if (block_r == x){
					A_coeff[indxA] += wG2 * w[s][RightPB_l[level][n][mu]][2*b+bet];// * SignR_l[level][n][mu];
					FLOPS += ca+cm;
				}
				//Only diff from zero when n+hat{mu} in Block(x+hat{mu})
				else if (block_r == RightPB_l[level+1][x][mu]){
					B_coeff[indxBC[mu]] += wG2 * w[s][RightPB_l[level][n][mu]][2*b+bet]; //Sign considered in the operator
					FLOPS += ca+cm;
				}
				//Only diff from zero when n-hat{mu} in Block(x)
				if (block_l == x){
					A_coeff[indxA] += wG3 * w[s][LeftPB_l[level][n][mu]][2*b+bet];// *  SignL_l[level][n][mu];
					FLOPS += ca+cm;
				}
				//Only diff from zero when n-hat{mu} in Block(x-hat{mu})
				else if (block_l == LeftPB_l[level+1][x][mu]){
					C_coeff[indxBC[mu]] += wG3 * w[s][LeftPB_l[level][n][mu]][2*b+bet];
					FLOPS += ca+cm;
				}
	
			}
			}	
			}
		}
	//---------Close loops---------//
	} //s
	} //p
	} //bet
	} //alf
	} //x 
	
}
*/
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
		xini = x_elements*bx; xfin = xini + x_elements;
		tini = t_elements*bt; tfin = tini + t_elements; 
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
					indx = x*Nt*colors*2 + t*colors*2 + c*2 + s;
					dot_product += std::conj(tvec.val[indx*Ntest+nt]) * tvec.val[indx*Ntest+ntt]; //v_nt . v_ntt
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
