"""Print the sacreBLEU and BERTScore configuration in use by
``scripts/eval/evaluate_all_models.py`` so the values can be pasted into
§5.3.1 of the thesis.

The thesis numbers in ``results/benchmarks/thesis_final_results.csv`` were
produced by ``hf_evaluate.load("sacrebleu")`` and ``hf_evaluate.load(
"bertscore")`` with no kwargs — i.e. all package defaults. This script makes
the implicit defaults explicit (sacreBLEU signature string + bert_score
language→model mapping + rescale_with_baseline default) and reports the
installed package versions, so the §5.3.1 ``[需核实]`` placeholders can be
filled with verifiable strings.

Run on the cluster env where ``sacrebleu`` / ``bert_score`` are installed
(local thesis env does not have them):

    python scripts/tools/print_metric_signatures.py

The script tries to feed sacreBLEU a real (preds, refs) pair from one of the
existing stage2 outputs so the signature reflects an actual corpus call; if
no such file is found it falls back to a 3-sentence toy sample (signature is
identical either way).
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_sample_stage2_csv() -> Path | None:
    """Return the first stage2.csv we can find under results/stage2/, or None."""
    root = Path(__file__).resolve().parent.parent.parent
    candidates = sorted((root / "results" / "stage2").rglob("stage2.csv"))
    return candidates[0] if candidates else None


def _load_preds_refs() -> tuple[list[str], list[str], str]:
    """Return (predictions, references, source_label).

    Falls back to a 3-sentence toy sample if no stage2 CSV is on disk.
    """
    csv = _find_sample_stage2_csv()
    if csv is not None:
        try:
            import pandas as pd
            df = pd.read_csv(csv)
            preds = df["text_gen"].fillna("").astype(str).tolist()
            refs = df["text_gt"].fillna("").astype(str).tolist()
            return preds, refs, str(csv)
        except Exception as e:
            print(f"  [warn] could not read {csv}: {e}", file=sys.stderr)
    toy_preds = [
        "Ein Mann öffnet die Tür.",
        "Sie geht zum Fenster.",
        "Er lächelt.",
    ]
    toy_refs = [
        "Der Mann öffnet die Tür.",
        "Anna geht zum Fenster.",
        "Viktor lächelt kurz.",
    ]
    return toy_preds, toy_refs, "<toy 3-sentence sample>"


def print_sacrebleu_signature() -> None:
    print("=" * 72)
    print("sacreBLEU")
    print("=" * 72)
    try:
        import sacrebleu
    except ImportError:
        print("  sacrebleu not installed in this env — skipped.")
        return
    print(f"  version: {sacrebleu.__version__}")
    preds, refs, source = _load_preds_refs()
    print(f"  sample:  {source}  ({len(preds)} segments)")
    metric = sacrebleu.BLEU()
    score = metric.corpus_score(preds, [refs])
    print(f"  score on sample:  {score.score:.2f}")
    print()
    print("  Paste into §5.3.1:")
    print(f"    {metric.get_signature()}")
    print()


def print_bertscore_signature() -> None:
    """Read bert_score's __version__ + lang2model from the source files
    *without* importing the package, which would pull in torch and OOM on a
    login node. Both values live in pure-Python text we can parse statically.
    """
    print("=" * 72)
    print("BERTScore")
    print("=" * 72)
    import importlib.util
    import re

    spec = importlib.util.find_spec("bert_score")
    if spec is None or spec.origin is None:
        print("  bert_score not installed in this env — skipped.")
        return
    pkg_dir = Path(spec.origin).parent
    init_txt = (pkg_dir / "__init__.py").read_text(encoding="utf-8")
    utils_txt = (pkg_dir / "utils.py").read_text(encoding="utf-8")

    m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", init_txt)
    version = m.group(1) if m else "<unknown>"
    print(f"  version: {version}")

    # bert_score's lang2model is a defaultdict whose default value is a
    # multilingual model; explicit per-language overrides are added by .update.
    # We extract:
    #   (a) the defaultdict default (`lambda: "<model>"`)
    #   (b) any explicit `"de": "<model>"` / `"it": "<model>"` entries
    default_m = re.search(
        r"lang2model\s*=\s*defaultdict\(\s*lambda\s*:\s*['\"]([^'\"]+)['\"]",
        utils_txt,
    )
    default_model = default_m.group(1) if default_m else "<unknown>"
    de_m = re.search(r"['\"]de['\"]\s*:\s*['\"]([^'\"]+)['\"]", utils_txt)
    it_m = re.search(r"['\"]it['\"]\s*:\s*['\"]([^'\"]+)['\"]", utils_txt)
    de_model = de_m.group(1) if de_m else f"{default_model}  (defaultdict fallback)"
    it_model = it_m.group(1) if it_m else f"{default_model}  (defaultdict fallback)"
    print(f"  lang='de' → model: {de_model}")
    print(f"  lang='it' → model: {it_model}")
    print(f"  rescale_with_baseline default: False  (evaluate_all_models passes no kwargs)")
    print()
    print("  Paste into §5.3.1:")
    print(
        f"    bert_score v{version}; de → {de_model}; "
        f"it → {it_model}; rescale_with_baseline=False; idf=False"
    )
    print()


def main() -> None:
    print_sacrebleu_signature()
    print_bertscore_signature()


if __name__ == "__main__":
    main()
