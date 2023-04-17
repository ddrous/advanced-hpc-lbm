#!/bin/bash

#SBATCH --job-name test_aai
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=1
#SBATCH --time 0:010:00
#SBATCH --partition=teach_cpu
#SBATCH --account=COSC028844
#SBATCH --output test_aai.out
######SBATCH --exclusive

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOB_ID
echo This job runs on the following machines:
echo `echo $SLURM_JOB_NODELIST | uniq`

#! Run the executable

time hostname

# mpirun -n 1  --bind-to socket ./d2q9-bgk input_128x128.params obstacles_128x128.dat
#mpirun -n 4 ./d2q9-bgk input_128x128.params obstacles_128x128.dat
#./d2q9-bgk input_128x256.params obstacles_128x256.dat
# mpiexec -n 1 ./d2q9-bgk input_256x256.params obstacles_256x256.dat
# ./d2q9-bgk input_1024x1024.params obstacles_1024x1024.dat
