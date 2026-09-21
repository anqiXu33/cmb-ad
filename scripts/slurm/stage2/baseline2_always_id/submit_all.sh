#!/bin/bash
# Submit all Baseline 2 (Always-ID) jobs
cd /home/anxu/shot-by-shot
mkdir -p logs/slurm/stage2/baseline2_always_id/

sbatch scripts/slurm/stage2/baseline2_always_id/run_8_Tage_im_August.sh
sbatch scripts/slurm/stage2/baseline2_always_id/run_Moskau_einfach.sh
sbatch scripts/slurm/stage2/baseline2_always_id/run_Baghdad_in_my_Shadow.sh
sbatch scripts/slurm/stage2/baseline2_always_id/run_Trommelwirbel.sh
sbatch scripts/slurm/stage2/baseline2_always_id/run_La_Tentazione_Di_Esistere.sh
sbatch scripts/slurm/stage2/baseline2_always_id/run_Quanto_basta.sh
sbatch scripts/slurm/stage2/baseline2_always_id/run_Riders_of_justice.sh
sbatch scripts/slurm/stage2/baseline2_always_id/run_Stuerm.sh

echo "All Baseline 2 (Always-ID) jobs submitted. Check with: squeue -u $USER"
