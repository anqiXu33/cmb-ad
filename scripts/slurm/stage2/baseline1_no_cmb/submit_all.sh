#!/bin/bash
# Submit all Baseline 1 (No-CMB ablation) jobs
cd /home/anxu/shot-by-shot
mkdir -p logs/slurm/stage2/baseline1_no_cmb/

sbatch scripts/slurm/stage2/baseline1_no_cmb/run_8_Tage_im_August.sh
sbatch scripts/slurm/stage2/baseline1_no_cmb/run_Moskau_einfach.sh
sbatch scripts/slurm/stage2/baseline1_no_cmb/run_Baghdad_in_my_Shadow.sh
sbatch scripts/slurm/stage2/baseline1_no_cmb/run_Trommelwirbel.sh
sbatch scripts/slurm/stage2/baseline1_no_cmb/run_La_Tentazione_Di_Esistere.sh
sbatch scripts/slurm/stage2/baseline1_no_cmb/run_Quanto_basta.sh
sbatch scripts/slurm/stage2/baseline1_no_cmb/run_Riders_of_justice.sh
sbatch scripts/slurm/stage2/baseline1_no_cmb/run_Stuerm.sh

echo "All Baseline 1 (No-CMB) jobs submitted. Check with: squeue -u $USER"
