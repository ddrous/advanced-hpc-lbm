## A meta script to submit jobs on BC4

for (( nprocs = 1 ; nprocs < 113; nprocs++ )); do

    ## Buld the job submit script
    cat results/base_job_submit_1.txt > DELETEME.sh
    echo -e "#SBATCH --output results/$nprocs.out" >> DELETEME.sh
    cat results/base_job_submit_2.txt >> DELETEME.sh
    echo -e "mpirun -n $nprocs ./d2q9-bgk input_128x128.params obstacles_128x128.dat" >> DELETEME.sh

    ## Submit the job
    sbatch DELETEME.sh

done
