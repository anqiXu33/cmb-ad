# CMB-AD follow-up scaffolding (2026-10-24)

Two ways to apply to your `cmb-ad` repo:

    git am cmb-ad-followup-2026-10-24.patch        # one commit on top of b920b6f
    # or
    tar xzf cmb-ad-followup-2026-10-24.tar.gz      # files only

Reproduce (CPU, ~3 min):

    pip install spacy lightgbm pytest gender-guesser nltk
    python -m spacy download en_core_web_sm
    python -m spacy download it_core_news_sm
    python -m followup.silver_labels      # -> results/followup/silver_labels.csv, ads.csv
    python -m followup.features           # -> features.csv
    python -m followup.rule_planner       # -> rule_preds.csv
    python -m followup.train              # -> eval_summary.md (face-based labels)
    python -m followup.text_chains        # -> text_features.csv (transcript-only chains)
    python -m followup.train --variant text   # -> text_eval_summary.md
    python -m followup.zero_subject_it    # Italian zero-subject self-test
    pytest tests/ -q                      # 315 tests (300 are regression vs the thesis bank)

`stage2/swissAD_qwen3.py` needs no change: `CharacterMemoryBank(...)` defaults to `RulePlanner`.
To try a learned planner inside the pipeline, implement `ReferencePlanner.decide()` and pass
`planner=...` to the bank.

## Round 3: does reading the text help? (2026-10-24)

    python -m followup.text_models        # TF-IDF text models + stacking, ~3 min on CPU

Fine-tune a pretrained encoder on the MacBook (Apple GPU / MPS):

    pip install torch transformers lightgbm scikit-learn pandas
    python -m followup.finetune_encoder --folds 1      # quick look, fold 0 only (~5-8 min)
    python -m followup.finetune_encoder                # all 5 folds + stacking (~30-40 min)
    python -m followup.finetune_encoder --dry_run      # offline smoke test, tiny random model

Results land in results/followup/encoder_<model>_summary.md, same columns as
text_models_summary.md. Requires the files produced by `python -m followup.text_chains`
(text_features.csv, text_rule_preds.csv, text_ads.csv), which are included in the tarball.

## On Kaggle (free GPU)

Open `kaggle/finetune_encoder.ipynb` on Kaggle (Create → Notebook → File → Import notebook,
or import from GitHub), set Accelerator = GPU and Internet = On, optionally add a
`GITHUB_TOKEN` secret, then Run all. The notebook clones the `followup` branch,
rebuilds the data, trains with `--fp16`, and pushes the summary back to the branch.

## Name heard in dialogue (REFRAMED, Kaggle, CPU)

Open `kaggle/name_heard_reframed.ipynb` on Kaggle (Accelerator None, Internet On), Run all.
It downloads REFRAMED train+validation (CC BY-NC 4.0), runs `python -m followup.name_heard`,
and pushes only `results/followup/name_heard_summary.md` / `.json` back to this branch.

## LLM experiments (Kaggle, GPU T4)

Open `kaggle/llm_experiments.ipynb` on Kaggle (GPU T4 x2, Internet On, GITHUB_TOKEN secret), Save & Run All.
- `followup/llm_zeroshot.py`: Qwen2.5-7B-Instruct (4-bit), no training, NAME vs PRONOUN from next-token logits.
- `followup/llm_finetune.py`: Qwen2.5-1.5B QLoRA classifier, 10 previous AD lines, fold 0.
Both compare against the rule, lgbm+role and stack+role on the same rows, overall and per syntactic position.
Offline smoke tests: `--dry_run`.
