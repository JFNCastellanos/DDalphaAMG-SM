#ifndef METHODS_H
#define METHODS_H

#include "amg.h"
#include "conjugate_gradient.h"

//Class for comparing different matrix-inversion methods.
class Methods {
public:
    Methods(const spinor& U, const spinor& rhs, const spinor& x0 ,const double m0, const double tol): 
    U(U), rhs(rhs), x0(x0), m0(m0),tol(tol){
        //Solution buffers
        xBiCG       = new spinor(mpi::maxSizeH);
        xCG         = new spinor(mpi::maxSizeH);
        xGMRES      = new spinor(mpi::maxSizeH);
        xSAP        = new spinor(mpi::maxSizeH);
        xFGMRES_AMG     = new spinor(mpi::maxSizeH);
        xFGMRES_SAP = new spinor(mpi::maxSizeH);
        xVcycle     = new spinor(mpi::maxSizeH);
    }
    ~Methods(){
        delete xBiCG;
        delete xCG;
        delete xGMRES;
        delete xSAP;
        delete xFGMRES_AMG;
        delete xFGMRES_SAP;
        delete xVcycle;
    }

    void BiCG(const int max_it,const bool print);
    void GMRES(const int len, const int restarts,const bool print);
    void CG(const bool print);
    void SAP(const int iterations,const int xblocks, const int tblocks,const bool print);
    void FGMRES_sap(const int len, const int restarts,const bool print);
    
    //int fgmresAMG(spinor& x, const bool print);
    void Vcycle(const int iterations,const bool print);
    
    //void check_solution(const spinor& x_sol);

    spinor* xBiCG;
    spinor* xCG;
    spinor* xGMRES;
    spinor* xSAP;
    spinor* xFGMRES_AMG;
    spinor* xFGMRES_SAP;
    spinor* xVcycle;

private:

    const spinor U;
    const spinor rhs;
    const spinor x0;
    const double m0;
    const double tol;
    double start, end;

};


#endif