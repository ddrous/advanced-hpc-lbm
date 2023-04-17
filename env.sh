# Add any `module load` or `export` commands that your code needs to
# compile and run to this file.

# module load languages/anaconda2/5.0.1
# module load languages/intel/2020-u4
# module load GCC/7.2.0-2.29
# module load OpenMPI/3.0.0-GCC-7.2.0-2.29
module load iimpi/2017.01-GCC-5.4.0-2.26



#### ---------- FOR FLAT MPI ----------###
export OMP_NUM_THREADS=1


#### ---------- FOR HYBRID OPEMMP+MPI ----------###
# export I_MPI_PIN_DOMAIN=omp
# # export I_MPI_PIN_DOMAIN=socket

# export OMP_PLACES=cores
# export OMP_PROC_BIND=true
# export OMP_NUM_THREADS=28
