#!/bin/bash
# Cron-driven m-* union health check for the 12 CMB-AD + B1 jobs
# (2662448-2662459). Compares the "Rewrote pred_ids with m-* union"
# line in each job's log to the [SIM] number from check_pred_ids_dtypes.py
# captured 2026-05-02. Emails on first mismatch, then auto-disables once all
# 12 jobs are verified OK.
#
# Email-only — does NOT auto-scancel. Operator decides next steps.

set -uo pipefail

EMAIL="anqixu528@gmail.com"
ROOT="/home/anxu/shot-by-shot"
STATE="$ROOT/results/_cron_mstar_check.state"
LOG="$ROOT/results/_cron_mstar_check.log"

mkdir -p "$ROOT/results"
touch "$STATE"

declare -A EXP_N EXP_T
EXP_N[8_Tage_im_August]=832;        EXP_T[8_Tage_im_August]=1039
EXP_N[Baghdad_in_my_Shadow]=609;    EXP_T[Baghdad_in_my_Shadow]=731
EXP_N[Trommelwirbel]=375;           EXP_T[Trommelwirbel]=420
EXP_N[La_Tentazione_Di_Esistere]=397; EXP_T[La_Tentazione_Di_Esistere]=521
EXP_N[Riders_of_justice]=427;       EXP_T[Riders_of_justice]=526
EXP_N[Stuerm]=417;                  EXP_T[Stuerm]=633

JOBS=(2662448 2662449 2662450 2662451 2662452 2662453 2662454 2662455 2662456 2662457 2662458 2662459)
mismatch_msg=""

for jobid in "${JOBS[@]}"; do
    grep -q "^${jobid}:" "$STATE" && continue

    info=$(sacct -j "$jobid" --format=JobName%64,State -P -n -X 2>/dev/null | head -1)
    [ -z "$info" ] && continue

    name=$(echo "$info" | cut -d'|' -f1 | tr -d ' ')
    state=$(echo "$info" | cut -d'|' -f2 | tr -d ' ')

    case "$state" in RUNNING|COMPLETED|TIMEOUT) ;; *) continue ;; esac

    if [[ "$name" == P3_27B_* ]]; then
        cond=cmb_ad; movie="${name#P3_27B_}"
    elif [[ "$name" == B1_NoCMB_* ]]; then
        cond=baseline1_no_cmb; movie="${name#B1_NoCMB_}"
    else
        continue
    fi

    log="$ROOT/logs/slurm/stage2/${cond}/log_${movie}_${jobid}.log"
    [ -f "$log" ] || { log=$(find "$ROOT/logs/slurm/stage2/${cond}/" -name "*${jobid}*.log" 2>/dev/null | head -1); }
    [ -f "$log" ] || continue

    line=$(grep "Rewrote pred_ids with m-\* union" "$log" 2>/dev/null | head -1)
    [ -z "$line" ] && continue   # job is RUNNING but hasn't reached rewrite yet

    pair=$(echo "$line" | grep -oE '[0-9]+/[0-9]+' | head -1)
    actual_n=${pair%/*}
    actual_t=${pair#*/}

    exp_n="${EXP_N[$movie]:-}"
    exp_t="${EXP_T[$movie]:-}"
    if [ -z "$exp_n" ]; then
        echo "$(date -Iseconds) WARN $jobid ($name): unknown movie '$movie'" >> "$LOG"
        continue
    fi

    if [ "$actual_n" = "$exp_n" ] && [ "$actual_t" = "$exp_t" ]; then
        echo "$(date -Iseconds) OK   $jobid $name: $actual_n/$actual_t == expected" >> "$LOG"
        echo "${jobid}:OK:${name}:${actual_n}/${actual_t}" >> "$STATE"
    else
        msg="$jobid $name: got $actual_n/$actual_t, expected $exp_n/$exp_t"
        echo "$(date -Iseconds) FAIL $msg" >> "$LOG"
        echo "${jobid}:MISMATCH:${name}:${actual_n}/${actual_t}:expected_${exp_n}/${exp_t}" >> "$STATE"
        mismatch_msg+=$'\n'"  - $msg"
    fi
done

if [ -n "$mismatch_msg" ]; then
    {
        echo "[thesis-slurm] m-* union health-check MISMATCH"
        echo
        echo "One or more CMB-AD/B1 jobs report a pred_ids m-* union count that"
        echo "disagrees with the pre-flight check_pred_ids_dtypes.py [SIM] number."
        echo
        echo "Mismatches this cycle:$mismatch_msg"
        echo
        echo "Next steps:"
        echo "  ssh uzh-cluster"
        echo "  cat $LOG"
        echo "  squeue -u anxu"
        echo "  # Decide: scancel <jobid> if confirmed broken"
    } | mail -s "[thesis-slurm] m-* union MISMATCH" "$EMAIL" \
        && echo "$(date -Iseconds) MAIL sent ($EMAIL)" >> "$LOG" \
        || echo "$(date -Iseconds) MAIL failed" >> "$LOG"
fi

ok_count=$(grep -c ":OK:" "$STATE" 2>/dev/null) || ok_count=0
if [ "$ok_count" -ge 12 ] && ! grep -q "^DONE$" "$STATE"; then
    echo "DONE" >> "$STATE"
    {
        echo "All 12 CMB-AD + B1 jobs verified OK on m-* union health-check."
        echo
        echo "OK list:"
        grep ":OK:" "$STATE"
        echo
        echo "Cron is now self-disabled. Remove the entry with:"
        echo "  ssh uzh-cluster 'crontab -l | grep -v check_mstar_union | crontab -'"
    } | mail -s "[thesis-slurm] m-* union ALL OK — cron done" "$EMAIL" \
        && echo "$(date -Iseconds) DONE mail sent" >> "$LOG"
fi
