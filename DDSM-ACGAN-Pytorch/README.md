# DDSM-ACGAN-Pytorch

Applies the ACGAN + VGG16 architecture reconstructed for [CovidGAN](../CovidGAN-Pytorch) to a
different domain and task: **benign vs malignant classification of CBIS-DDSM mammography ROI
patches**, augmented with GAN-synthesized ROIs.

This exists because a survey of GAN-augmentation papers ([`GAN_Medical_Imaging_Paper_Survey.md`](../GAN_Medical_Imaging_Paper_Survey.md))
found that CovidGAN's original task (COVID vs Normal CXR) is too easy for a modern CNN to
benefit measurably from synthetic augmentation — the classifier saturates before the GAN can
help. Benign vs malignant mammography classification is a genuinely harder baseline, so it's a
better test of whether the augmentation technique itself does anything.

## Architecture

Identical generator/discriminator/classifier to CovidGAN (see `ddsm_acgan/models.py`) — it's a
generic conditional-GAN + frozen-VGG16-head setup, not tied to chest X-rays. Only the data
pipeline is domain-specific.

## Data

Two things are needed, both expected to already exist on Google Drive (see `DDSM_Colab.ipynb`):

1. **Pre-extracted ROI patches**, one PNG per abnormality, named following the standard
   CBIS-DDSM convention: `P_<patient_id>_<LEFT|RIGHT>_<CC|MLO>_<mass|calcification>_<abnormality_id>.png`
   (e.g. `P_01152_RIGHT_MLO_mass_3.png`).
2. **The four CBIS-DDSM case-description CSVs**: `mass_case_description_train_set.csv`,
   `mass_case_description_test_set.csv`, `calc_case_description_train_set.csv`,
   `calc_case_description_test_set.csv`. These carry the `pathology` column
   (`MALIGNANT` / `BENIGN` / `BENIGN_WITHOUT_CALLBACK`) and are the source of truth for both the
   label and the train/test split.

`prepare_ddsm.py` joins ROI filenames against the CSVs by
`(patient_id, side, view, abnormality_type, abnormality_id)` and **preserves the CBIS-DDSM's own
train/test assignment** rather than re-randomizing it — that's the standard benchmark protocol,
so results here are comparable to other CBIS-DDSM literature. Any ROI file that doesn't match a
CSV row (unrecognized pathology value, or a filename that doesn't parse) is dropped and reported.

**Always check the match-rate report `prepare_ddsm.py` prints before trusting the manifest** —
if it's low, the CSV column names or ROI filenames don't match what the script assumes; it prints
a sample of unmatched files to help diagnose.

## Setup

```bash
pip install -r requirements.txt   # torch/torchvision + scikit-learn, matplotlib, pillow
```

Or just use `DDSM_Colab.ipynb`, which handles Drive mounting, repo cloning, and dependency
install for you on a free GPU runtime.

## Pipeline

```bash
python prepare_ddsm.py \
    --roi-dir /path/to/ddsm_rois/rois \
    --mass-train-csv /path/to/mass_case_description_train_set.csv \
    --mass-test-csv  /path/to/mass_case_description_test_set.csv \
    --calc-train-csv /path/to/calc_case_description_train_set.csv \
    --calc-test-csv  /path/to/calc_case_description_test_set.csv \
    --out-dir data

python train_gan.py --manifest data/manifest.csv --out-dir runs/gan

python generate_synthetic.py --checkpoint runs/gan/checkpoints/ddsm_acgan_final.pt \
    --out-dir data/synthetic --n-benign 600 --n-malignant 600
#   adjust --n-benign/--n-malignant based on the class counts prepare_ddsm.py reports --
#   CBIS-DDSM isn't perfectly balanced, unlike CovidGAN's dataset

python train_classifier.py --manifest data/manifest.csv --mode ad --out-dir runs/cnn_ad
python train_classifier.py --manifest data/manifest.csv --mode sa \
    --synthetic-dir data/synthetic --out-dir runs/cnn_sa
```

Compare `runs/cnn_ad/metrics.txt` against `runs/cnn_sa/metrics.txt` — that comparison is the
actual experiment.

## Repo layout

```
ddsm_acgan/
  models.py     Generator, Discriminator, build_classifier (same architecture as CovidGAN)
  data.py       ROI filename parsing, CBIS-DDSM CSV join, manifest read/write, Dataset
  metrics.py    classification table, confusion matrix, PCA plot
prepare_ddsm.py        raw ROI folder + 4 CSVs -> data/manifest.csv, with match-rate diagnostics
train_gan.py            train the ACGAN
generate_synthetic.py   sample the trained generator into a synthetic pool
train_classifier.py     train + evaluate the detection CNN (AD or SA mode)
DDSM_Colab.ipynb         ready-to-run Colab notebook for the full pipeline on a free GPU
```

## Known limitation

`generate_synthetic.py`'s default 600/600 benign/malignant counts are arbitrary placeholders —
CBIS-DDSM's actual class balance depends on which ROIs you have and isn't known until
`prepare_ddsm.py` reports it. Check that output and adjust before generating the synthetic pool.
