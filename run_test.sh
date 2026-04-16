#!/bin/bash


N=512
COUNT=0
SAP_BLOCK_NUMBER=8
for BSIZE in 128 64 32 16 8 4; do
    NBLOCKS=$((N/BSIZE))
    for NV in 5 10; do
        printf "%d " 0 > parameters
        printf "%d " ${NBLOCKS} >> parameters
        printf "%d " ${NBLOCKS} >> parameters
        printf "%d " ${NV} >> parameters
        printf "%d " ${SAP_BLOCK_NUMBER} >> parameters
        printf "%d\n" ${SAP_BLOCK_NUMBER} >> parameters
        cd build 
        mpirun --oversubscribe -n 16 DDAlpha_${N}x${N} < inputs
        rm -rf results_${COUNT}
        mkdir -p results_${COUNT}
        mv *.dat results_${COUNT}
        COUNT=$((COUNT+1))
        cd ..
    done
done
