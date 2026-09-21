#!/bin/bash
# Resubmit the 18 non-pilot Stage-2 jobs (6 films x 3 conditions).
#
# Run this AFTER the 6 pilot jobs (Moskau_einfach + Quanto_basta x 3 conditions)
# have COMPLETED and human-eval extraction is unblocked.
#
# Skipped on purpose: Moskau_einfach (DE pilot), Quanto_basta (IT pilot).
# Their stage2.csv outputs already exist in results/stage2/{cmb_ad,baseline1_no_cmb,baseline2_always_id}/ -
# do NOT re-submit them or you will overwrite completed work.
cd /home/anxu/shot-by-shot
mkdir -p logs/slurm/stage2/cmb_ad/ logs/slurm/stage2/baseline1_no_cmb/ logs/slurm/stage2/baseline2_always_id/

FILMS=(
  8_Tage_im_August
  Baghdad_in_my_Shadow
  Trommelwirbel
  La_Tentazione_Di_Esistere
  Riders_of_justice
  Stuerm
)

for film in "${FILMS[@]}"; do
  sbatch scripts/slurm/stage2/cmb_ad/run_${film}.sh
  sbatch scripts/slurm/stage2/baseline1_no_cmb/run_${film}.sh
  sbatch scripts/slurm/stage2/baseline2_always_id/run_${film}.sh
done

echo "Submitted 18 jobs (6 films x 3 conditions). Check with: squeue -u $USER"
