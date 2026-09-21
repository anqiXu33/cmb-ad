# Side experiment: Calibrated query-bank A/B

**Status:** scripts ready, not yet run on cluster (as of 2026-04-27).

**TL;DR:** isolated, low-cost check of how often FocusedAD-style calibrated
centroids change `pred_ids` vs raw InsightFace headshot embeddings, on a
single film. Output is one CSV + one text report. **Decoupled from the
thesis main pipeline** — running this does not change any main-line
artefact.

---

## Why this is a side task, not a main contribution

The thesis main claim is the CharacterMemoryBank + SWISS-AD prompting
(`stage2/swissAD_qwen3.py`). That comparison (4 models × 8 movies) is fair
*as long as every model consumes the same `pred_ids`* — accuracy of
`pred_ids` itself is not the variable being studied.

`char_recog/build_query_bank.py` produces two artefacts:
- `results/preprocess/rep_crops/<imdbid>/*.jpg` — **used** by
  `char_recog/generate_char_descriptions.py` to write VLM physical
  descriptions into `imdb_chars_enriched.json`. Already wired into CMB
  via the REINTRODUCE flag. This is the live use-case.
- `results/preprocess/query_bank/query_bank_*.pkl` — calibrated 512-d
  centroids per actor. Designed to replace raw headshot embeddings in
  `char_recog/main.py --query_bank_path`, but **not currently invoked
  by any pipeline driver**. The 4 shards exist but feed nothing.

So the question this experiment answers: *if we did wire the calibrated
bank into `pred_ids` generation, would it actually change anything?*
Cheap to find out without re-running Stage 1/2.

---

## Decision rule

After running `run_moskau.sh` and inspecting `diff_report.txt`:

| `raw_rerun vs calibrated` changed clips | Action |
|------------------------------------------|--------|
| < 5%                                     | Drop the bank-as-`pred_ids` claim. Mention it as future work. Keep `rep_crops` as the live use-case. |
| 5–10%                                    | Mention as appendix-only ablation. Don't full-rerun. |
| ≥ 10%                                    | Consider full A/B: re-run `char_recog/main.py --query_bank_path` for all 8 movies, then re-run Stage 1 + Stage 2 + eval. Heavy. |

Inside the changed clips, also look at the per-clip diff sample at the
bottom of `diff_report.txt`: are the calibration-only IDs mostly fixing
known mistakes (e.g., a major character that headshot-mode often missed)
or introducing new noise?

---

## Files

| File | Role |
|------|------|
| `run_ab.py`        | Single-pass face detection per clip; matches same detected faces against both raw headshots and calibrated centroids. Outputs CSV with `pred_ids` (preserved from input) + `bboxes_rerun` + `pred_ids_raw_rerun` + `pred_ids_calibrated`. |
| `diff.py`          | Set-difference summary over the three pred-id columns. Prints per-pair stats (changed-clip ratio, Jaccard, top added/removed IDs) and a sample of changed clips. |
| `run_moskau.sh`    | SLURM driver — runs `run_ab.py` then `diff.py` on Moskau_einfach (`tt28132461`, in `query_bank_0.pkl`). |

Imports `process_video` and `group_and_filter_id` from `../../char_recog/utils.py`
(via `sys.path` injection in `run_ab.py`) — no duplication.

---

## How to run (cluster only — needs videos + InsightFace)

```bash
ssh anxu@cluster
cd /home/anxu/shot-by-shot
git pull   # bring in this folder
sbatch experiments/calibrated_query_bank_ab/run_moskau.sh
```

Outputs land under `experiments/calibrated_query_bank_ab/results/Moskau_einfach/`:
- `pred_ids_ab.csv` — full annotation rows + the four pred-id-related columns
- `diff_report.txt` — text summary

To extend to another film: copy `run_moskau.sh`, change `MOVIE`, `IMDB_ID`,
and `SHARD` (lookup: `python -c "import pickle; [print(i, list(pickle.load(open(f'results/preprocess/query_bank/query_bank_{i}.pkl','rb')).keys())) for i in range(4)]"`).

---

## Output columns explained

| Column | Source | Notes |
|--------|--------|-------|
| `pred_ids`             | from input annotation CSV (untouched) | original cluster run, raw headshot |
| `bboxes`               | from input annotation CSV (untouched) | original detected bboxes |
| `bboxes_rerun`         | new face detection in this run        | for spot-checking detection drift |
| `pred_ids_raw_rerun`   | raw headshot embedding × new faces    | apples-to-apples baseline |
| `pred_ids_calibrated`  | calibrated centroid × new faces       | apples-to-apples treatment |

Why `pred_ids_raw_rerun` matters: comparing it to `pred_ids_calibrated`
isolates pure calibration effect (same detections, only cosine reference
differs). Comparing original `pred_ids` to `pred_ids_calibrated` mixes
calibration with face-detection nondeterminism.

---

## Progress log

- **2026-04-27** — scripts authored and folded into this side directory.
  Cluster run pending. Decision deferred until `diff_report.txt` is in
  hand.
