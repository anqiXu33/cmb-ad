"""Post-thesis follow-up: learned reference-form selection for CMB-AD.

Modules
-------
mentions_en      spaCy-based English NAME / DESC / PRN mention extractor
silver_labels    per-(AD, character) reference-form labels from ground-truth AD
features         discourse-state features for the learned planner
rule_planner     re-implementation of the thesis rule planner on the same inputs
train            LR / LightGBM baselines vs. rule planner
zero_subject_it  Italian pro-drop (null-subject) detector for the IT evaluation
"""
