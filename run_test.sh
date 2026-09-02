#!/bin/bash

BETA=4
N=32   #Number of lattice sites on x or t
SAP_BLOCK_NUMBER=4 #Number of SAP blocks on the x and t direction on a local rank
BSIZE=8   #Blocks size
NBLOCKS=$((N/BSIZE)) #Number of blocks across the whole lattice
NV=10   #Number of test vectors
RANKS_X=2 #MPI ranks on the x direction
RANKS_T=2 #MPI ranks on the t direction
LEVELS=2 #Number of levels
TM=0.05 #twisted mass
if [ $TM -eq 0 ]; then
    TM_DIR="0"
else
    TM_DIR="${TM/0./0}"
fi
M0=-0.1033 #Bare mass parameters
MSTR="${M0/-0./-0}"
CONFPATH="tm_confs/b${BETA}_${N}x${N}/mu_${TM_DIR}/2D_U1_Ns${N}_Nt${N}_b40000_m${MSTR}_0.ctxt"
RHSPATH="rhs_conf0_${N}x${N}.rhs"
PARAMETERS_PATH="parameters"
CMAKELISTS="CMakeLists.txt"
COMPILE=0 #1 Compile code, anything different doesn't compile

#Two levels
if [ $LEVELS -eq 2 ]; then
    printf "%d " 0 > parameters
    printf "%d " ${NBLOCKS} >> parameters
    printf "%d " ${NBLOCKS} >> parameters
    printf "%d " ${NV} >> parameters
    printf "%d " ${SAP_BLOCK_NUMBER} >> parameters
    printf "%d\n" ${SAP_BLOCK_NUMBER} >> parameters
elif [ $LEVELS -eq 3 ]; then
    printf "%d " 0 > parameters
    printf "%d " ${NBLOCKS} >> parameters
    printf "%d " ${NBLOCKS} >> parameters
    printf "%d " ${NV} >> parameters
    printf "%d " ${SAP_BLOCK_NUMBER} >> parameters
    printf "%d\n" ${SAP_BLOCK_NUMBER} >> parameters

    printf "%d " 1 >> parameters
    printf "%d " 16 >> parameters
    printf "%d " 16 >> parameters
    printf "%d " 10 >> parameters
    printf "%d " 2 >> parameters
    printf "%d\n" 2 >> parameters
elif [ $LEVELS -eq 4 ]; then
    printf "%d " 0 > parameters
    printf "%d " ${NBLOCKS} >> parameters
    printf "%d " ${NBLOCKS} >> parameters
    printf "%d " ${NV} >> parameters
    printf "%d " ${SAP_BLOCK_NUMBER} >> parameters
    printf "%d\n" ${SAP_BLOCK_NUMBER} >> parameters

    printf "%d " 1 >> parameters
    printf "%d " 16 >> parameters
    printf "%d " 16 >> parameters
    printf "%d " 10 >> parameters
    printf "%d " 2 >> parameters
    printf "%d\n" 2 >> parameters

    printf "%d " 2 >> parameters
    printf "%d " 8 >> parameters
    printf "%d " 8 >> parameters
    printf "%d " 10 >> parameters
    printf "%d " 1 >> parameters
    printf "%d\n" 1 >> parameters
fi

#Inputs
printf "%d\n" ${RANKS_X} > inputs
printf "%d\n" ${RANKS_T} >> inputs
printf "%d\n" ${LEVELS} >> inputs
printf "%f\n" ${M0} >> inputs
printf "%f\n" ${TM} >> inputs
printf "%s\n" ${CONFPATH} >> inputs
printf "%s\n" ${RHSPATH} >> inputs
printf "%s\n" ${PARAMETERS_PATH} >> inputs 

if [ $COMPILE -eq 1 ]; then
    sed -i "23s/set(NS \".*\")/set(NS \"${N}\")/" "$CMAKELISTS"
    sed -i "24s/set(NT \".*\")/set(NT \"${N}\")/" "$CMAKELISTS"
    rm -rf build
    mkdir build
    cd build
    cmake ../
    make -j 24
    mv DDAlpha_${N}x${N} ../
    cd ../
fi

mpirun --oversubscribe -n $(($RANKS_T*$RANKS_X)) DDAlpha_${N}x${N} < inputs