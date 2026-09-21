#!/bin/bash

format_m0() {
    local m0="$1"
    local sign=""
    if [ "$(awk -v x="$m0" 'BEGIN{print (x<0)?1:0}')" -eq 1 ]; then
        sign="-"
    fi
    local digits
    digits=$(awk -v x="$m0" 'BEGIN{printf "%05.0f", (x<0?-x:x)*10000}')
    printf '%s%s\n' "$sign" "$digits"
}


BETA=4
N=512   #Number of lattice sites on x or t
SAP_BLOCK_NUMBER=4 #Number of SAP blocks on the x and t direction on a local rank
BSIZE=4   #Blocks size
NBLOCKS=$((N/BSIZE)) #Number of blocks across the whole lattice
NV=10   #Number of test vectors
RANKS_X=4 #MPI ranks on the x direction
RANKS_T=4 #MPI ranks on the t direction
LEVELS=3 #Number of levels
TM=0 #twisted mass
CSW=0.0 #clover term constant
M0=-0.1023 #-0.0461 #Bare mass parameters
MSTR=$(format_m0 "$M0")
TM_DIR=$(format_m0 "$TM")
BSTR=$(format_m0 "BETA")
CONFPATH="2D_U1_Ns512_Nt512_b40000_m-01023_0.ctxt"  #"clover/b${BETA}_${N}x${N}/tm_${TM_DIR}/m${MSTR}/2D_U1_Ns${N}_Nt${N}_b40000_m${MSTR}_0.ctxt"
RHSPATH="rhs_conf0_${N}x${N}.rhs"
PARAMETERS_PATH="parameters"
CMAKELISTS="CMakeLists.txt"
COMPILE=1 #1 Compile code, anything different doesn't compile

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
printf "%f\n" ${CSW} >> inputs
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