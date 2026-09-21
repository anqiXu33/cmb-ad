#!/bin/bash
# Submit both B-condition sanity-check jobs
cd /home/anxu/shot-by-shot
sbatch scripts/slurm/ablations/sanity_check_B/run_Baghdad_in_my_Shadow.sh
sbatch scripts/slurm/ablations/sanity_check_B/run_Quanto_basta.sh
echo "Submitted B-condition jobs for Baghdad + Quanto_basta"
