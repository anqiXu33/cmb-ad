#!/bin/bash
# Submit all CMB-AD (Qwen3.5-27B) jobs
cd /home/anxu/shot-by-shot
mkdir -p logs/slurm/stage2/cmb_ad/

sbatch scripts/slurm/stage2/cmb_ad/run_8_Tage_im_August.sh
sbatch scripts/slurm/stage2/cmb_ad/run_Moskau_einfach.sh
sbatch scripts/slurm/stage2/cmb_ad/run_Baghdad_in_my_Shadow.sh
sbatch scripts/slurm/stage2/cmb_ad/run_Trommelwirbel.sh
sbatch scripts/slurm/stage2/cmb_ad/run_La_Tentazione_Di_Esistere.sh
sbatch scripts/slurm/stage2/cmb_ad/run_Quanto_basta.sh
sbatch scripts/slurm/stage2/cmb_ad/run_Riders_of_justice.sh
sbatch scripts/slurm/stage2/cmb_ad/run_Stuerm.sh

echo "All jobs submitted. Check with: squeue -u $USER"
