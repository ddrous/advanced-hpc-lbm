# Add any `module load` or `export` commands that your code needs to
# compile and run to this file.

# module load languages/anaconda2/5.0.1
# module load languages/intel/2020-u4
# module load GCC/7.2.0-2.29
# module load OpenMPI/3.0.0-GCC-7.2.0-2.29
module load iimpi/2017.01-GCC-5.4.0-2.26

# https://vlaams-supercomputing-centrum-vscdocumentation.readthedocs-hosted.com/en/latest/software/hybrid_mpi_openmp_programs.html
# export I_MPI_PIN_DOMAIN=omp       ## On BC 4, optimal usage might be 8 ranks with 14 threads per ranks (cuz of NUMA)
export I_MPI_PIN_DOMAIN=socket

export OMP_PLACES=cores
export OMP_PROC_BIND=true
export OMP_NUM_THREADS=28
