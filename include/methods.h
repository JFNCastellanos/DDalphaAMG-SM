#ifndef METHODS_H
#define METHODS_H

#include "amg.h"
#include "conjugate_gradient.h"

//Class for comparing different matrix-inversion methods.
class Methods {
public:
    Methods(const spinor& U, const spinor& rhs, const spinor& x0 ,const double m0): 
    U(U), rhs(rhs), x0(x0), m0(m0){

    }

    void BiCG(spinor& x, const int max_it,const bool print);
    void GMRES(spinor& x, const int len, const int restarts,const bool print);
    void CG(spinor& x);
    void FGMRES_sap(spinor &x,const bool print);
    void SAP(spinor& x,const int iterations,const bool print);
    //int fgmresAMG(spinor& x, const bool print);
    void multigrid(spinor& x, const bool print);
    
    void check_solution(const spinor& x_sol);

private:

    const spinor U;
    const spinor rhs;
    const spinor x0;
    const double m0;
    clock_t start, end;
    double elapsed_time;
    double startT, endT;



};


#endif