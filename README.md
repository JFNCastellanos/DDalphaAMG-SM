# Domain Decomposition Adaptive Algebraic Multigrid for the Schwinger Model

The code provides an MPI implementation of the solver DD$\alpha$AMG for the Schwinger model with the Wilson discretization, where the grid is decomposed into different 2D domains which belong to a different core. A short summary of this method: Schwarz Alternating Procedure (SAP), a block iterative method, is used a smoother. The interpolator is built with *test vectors* that capture the near-kernel of the Dirac matrix. This vectors are computed during a setup phase. As a coarse-grid solver we use GMRES without preconditioning. As an outer solver we utilize FGMRES. Both the V-cycle and K-cycle are available. The software is only for testing the solvers and is not prepared for an integration with an HMC simulation. 


To compile the solver do 

```
mkdir build
cd build
cmake ../
make
```

This has to be done for a specific lattice size, which can be modified in `CMakeLists.txt`. To choose the number of levels, size of aggregates, number of test vectors and size of SAP blocks modify the `parameters` file, which is read during runtime. The layout is

$$l, A_x, A_t, N_v, B_x, B_t$$

where 

* $l$ level number (starting from 0).
* $A_i$ number of aggregates on $i$-direction (across the whole lattice).
* $N_v$ number of test vectors to coarses the current level.
* $B_i$ number of blocks on $i$-direction for SAP (for a local lattice. *i.e.* for a single process).

Other solver parameters, such as the number of pre and pos-smoothing steps, solver tolerance, GMRES length, etc are fixed during compilation. They can be modified in `src/variables.cpp`. 

The code inverts the Dirac matrix for a given gauge configuration with several methods to compare their performance. These methods are called in the `main` file. Those methods which you do not want to try just comment them out. Gauge configurations have to be in the format corresponding to this Schwinger Model simulation https://github.com/Fabian2598/SchwingerModel. A rhs must be provided as well. The format for this is exactly the same as with the gauge configurations.

When running the code the program asks for 

```
mpi ranks on x
mpi ranks on t
number of levels
m0 (bare mass parameter)
path to the configuration file
path to a rhs file
path to parameters file
```

We provide a script portraying an example for a configuration of $V=512^2$ with $m_0=-0.1023, |m_0-m_c|\sim 0.0001$.

run_test.sh





