# #!/bin/bash
# # Load STREAM build with AOCC into environment
# # NOTE: if you have compiled multiple versions you may need to be more specific
# # Spack will complain if your request is ambiguous and could refer to multiple
# # packages. (https://spack.readthedocs.io/en/latest/basic_usage.html#ambiguous-specs)
# spack load stream %aocc

# # For optimal stream performance, it is recommended to set the following OS parameters (requires root/sudo access)
# echo always > /sys/kernel/mm/transparent_hugepage/enabled     # Enable hugepages
# echo always > /sys/kernel/mm/transparent_hugepage/defrag     # Enable hugepages
# echo 3 > /proc/sys/vm/drop_caches                            # Clear caches to maximize available RAM
# echo 1 > /proc/sys/vm/compact_memory                         # Rearrange RAM usage to maximise the size of free blocks

# # Optimize OpenMP performance behavious
# export OMP_SCHEDULE=static  # Disable dynamic loop scheduling
# export OMP_PROC_BIND=TRUE   # Bind threads to specific resources
# export OMP_DYNAMIC=false    # Disable dynamic thread pool sizing

# # OMP_PLACES is used for binding OpenMP threads to cores
# # See: https://www.openmp.org/spec-html/5.0/openmpse53.html

# ############# FOR AMD EPYC™ 9654 ##################
# # For example, a dual socket AMD 4th Gen EPYC™ Processor with 192 (96x2) cores,
# # with 4 threads per L3 cache: 96 total places, stride by 2 cores:
# export OMP_PLACES=0:96:2
# export OMP_NUM_THREADS=96

# # ############# FOR AMD EPYC™ 9755 ##################
# # # For example, a dual socket AMD 5th Gen EPYC™ Processor with 256 (128x2) cores,
# # # with 1 thread per L3 cache: 32 total places, stride by 8 cores:
# export OMP_PLACES=0:32:8
# export OMP_NUM_THREADS=32






## For the HPC lab
# Optimize OpenMP performance behavious
# export OMP_SCHEDULE=static  # Disable dynamic loop scheduling
export OMP_NUM_THREADS=14
export OMP_PLACES=cores
export OMP_PROC_BIND=close
numactl -N 0 - m 0

# Running stream
./stream_c.exe
