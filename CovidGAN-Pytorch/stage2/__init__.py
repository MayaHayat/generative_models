"""
Stage 2 -- Architecture improvement (kept fully separate from the Stage 1
reconstruction).

Everything under this package is NEW code for the "Improve the Architecture"
half of the project. Stage 1 files (covidgan/, train_classifier.py, etc.) are
imported for their data/metrics utilities but are never modified, so the
reconstruction and the improvement stay easy to tell apart.

The improvement itself lives in `stage2/model.py`; the training/evaluation
driver is `stage2/train_stage2.py`; `stage2/compare_results.py` assembles the
paper / reconstruction / improved comparison table.
"""
