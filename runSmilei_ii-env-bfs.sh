#!/bin/bash 
source ~/env_skl.sh

#MSUB -r smilei_mpi
#MSUB -n 10
#MSUB -c 24
#MSUB -A gen10062
#MSUB -q skylake
#MSUB -T 3600
#MSUB -m work,scratch

export OMP_NUM_THREADS=24
export OMP_SCHEDULE=dynamic
export OMP_PROC_BIND=true

ccc_mprun ./smilei LWFA_ii-env-bfs-2.py 
