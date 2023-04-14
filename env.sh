# Add any `module load` or `export` commands that your code needs to
# compile and run to this file.

module load languages/anaconda2/5.0.1
module load languages/intel/2020-u4
module load GCC/7.2.0-2.29
module load OpenMPI/3.0.0-GCC-7.2.0-2.29

export OMP_PLACES=cores
export OMP_PROC_BIND=true
export OMP_NUM_THREADS=1
