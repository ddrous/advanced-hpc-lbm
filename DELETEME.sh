#!/bin/bash

#SBATCH --job-name d2q9-bgk
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 28
#SBATCH --time 3:00:00
#SBATCH --partition=teach_cpu
#SBATCH --account=COSC028844
#SBATCH --exclusive
#SBATCH --output results/112.out

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOB_ID
echo This job runs on the following machines:
echo `echo $SLURM_JOB_NODELIST | uniq`

mpirun -n 112 ./d2q9-bgk input_128x128.params obstacles_128x128.dat
