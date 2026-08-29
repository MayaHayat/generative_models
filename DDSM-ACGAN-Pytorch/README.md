# DDSM-ACGAN-Pytorch

Applies the ACGAN + VGG16 architecture reconstructed for [CovidGAN](../CovidGAN-Pytorch) to a
different domain and task: **benign vs malignant classification of CBIS-DDSM mammography ROI
patches**, augmented with GAN-synthesized ROIs.

This exists as the project's **second-dataset validation**. The CovidGAN reconstruction could not
reproduce the paper's +10-point augmentation lift, and the leading suspicion was the COVID data
itself: the Kaggle COVID-19 Radiography Database has been re-released several times since the
March-2020 snapshot the paper used and is now larger and cleaner, so the baseline sits near ceiling
with little augmentation headroom left. But that is a claim about *that particular dataset*, not
about the method. The way to test it is to run the whole pipeline again, unchanged, on a completely
independent dataset and check whether each conclusion recurs. CBIS-DDSM is static and versioned, so
any effect here is attributable to the model, not to dataset drift.

A secondary benefit of this dataset: benign vs malignant mammography is a genuinely harder task than
COVID vs Normal CXR, so its baseline sits well below ceiling. That is what gives it the headroom to show augmentation
working at all, where COVID's near-saturated classifier left almost none — useful, but a property
that makes it a good validation target, not the reason it was chosen.

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
pip install -r requirements.txt
```

`torch`, `torchvision`, `numpy`, `pillow`, `scikit-learn`, `matplotlib`, `wandb` (run logging), and
`torchmetrics` for the FID/KID metrics in `fid_kid_eval.py`.

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
  models.py             Generator, Discriminator, build_classifier (same architecture as CovidGAN)
  models_improved.py    spectral-norm architecture variant
  diffaugment.py        differentiable augmentation
  data.py               ROI filename parsing, CBIS-DDSM CSV join, manifest read/write, Dataset
  metrics.py            classification table, confusion matrix, PCA plot

prepare_ddsm.py         raw ROI folder + 4 CSVs -> data/manifest.csv, with match-rate diagnostics
train_gan.py            train the ACGAN (--improved for the spectral-norm variant, --seed, --resume)
generate_synthetic.py   sample the trained generator into a synthetic pool
train_classifier.py     train + evaluate the detection CNN (AD or SA mode)
multiseed.py            multi-seed AD-vs-SA comparison
data_scarcity.py        real-data subsampling sweep
fid_kid_eval.py         manifest-based FID/KID for a synthetic pool
realness_probe.py       how separable synthetic images are from real ones
make_figures.py         regenerates the figures used in the report

DDSM_Colab.ipynb        ready-to-run Colab notebook for the full pipeline on a free GPU
DDSM_Colab_run.ipynb    executed run (seed 0), outputs retained
DDSM_Colab-seed1run.ipynb  executed run (seed 1), outputs retained
```

The written report is not in this repository; it is submitted separately. Generated artifacts
(`data/`, `runs/`, weights) are gitignored and reproducible from the commands above. Run metrics
were logged to Weights & Biases.

## Known limitation

`generate_synthetic.py`'s default 600/600 benign/malignant counts are arbitrary placeholders —
CBIS-DDSM's actual class balance depends on which ROIs you have and isn't known until
`prepare_ddsm.py` reports it. Check that output and adjust before generating the synthetic pool.
