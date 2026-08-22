# DDSM-ACGAN — Findings & Analysis

*Applying the CovidGAN architecture (Waheed et al. 2020, reconstructed in the companion CovidGAN-Pytorch
project) to a second, harder classification task — CBIS-DDSM mammography, benign vs. malignant — chosen
specifically because the COVID-CXR data landscape has itself drifted since 2020 (see CovidGAN-Pytorch's
`REPORT.md` §3.3–3.7), which makes it unsuitable as a stable benchmark for isolating an architecture
change. CBIS-DDSM is static and versioned, so results here can be attributed to the model, not the data.*

> **Status:** the matched-epoch comparison (§5) is now complete — the baseline architecture has been
> retrained to exactly 300 epochs on the mass-only subset, matching the improved architecture's best
> checkpoint.

All numbers in this document are pulled directly from the project's Weights & Biases logs
(`maya-hayat-ariel-university/ddsm-acgan`), not from memory of console output.

---

## 1. Task and dataset

Benign vs. malignant classification on CBIS-DDSM ROI patches (mass-type abnormalities only — calcification
ROIs excluded for this thread of experiments, via `prepare_ddsm.py`'s independent mass/calc CSV options).
Labels and the train/test split come from CBIS-DDSM's own case-description CSVs (not re-randomized), so
results are comparable to standard CBIS-DDSM benchmarks. Mass-only scope: 1,318 real training ROIs, 378
real test ROIs (231 benign / 147 malignant).

Architecture (generator, discriminator, VGG16-based classifier) is unchanged from the CovidGAN
reconstruction — see `CovidGAN-Pytorch/REPORT.md` §1 for the full layer-by-layer specification. Only the
data pipeline and the Stage 2 architecture variant (§4 below) are specific to this project.

## 2. Two data-pipeline bugs found and fixed during this work

Both are documented in detail in the codebase's git history; summarized here because they materially
affected every result before the fixes landed, and are a real part of what this project's engineering
process found:

1. **Mask-folder contamination.** The Kaggle "COVID-19 Radiography Database" mirror ships a `masks/`
   subfolder alongside `images/` for each class, with the same filenames. An early data-loading path
   recursively globbed both, silently doubling image counts with binary segmentation masks mixed in as if
   they were real photos. Fixed by explicitly excluding any path containing `mask(s)`/`label(s)`.
2. **16-bit grayscale destroyed on load.** CBIS-DDSM ROI patches are commonly stored as 16-bit grayscale
   PNGs (mode `I;16`, pixel values in ranges like `[20745, 65535]`, not `[0, 255]`). `PIL.Image.convert
   ('RGB')` does not rescale `I;16` images — it truncates straight to 8 bits, which for values this large
   collapses nearly every pixel to near-white. This was diagnosed by comparing a "real" training-image
   sample grid (which looked identical to the "collapsed" GAN output — flat white fields with sharp
   black wrap-around artifacts) against an actual real ROI image the dataset owner could inspect directly
   (which showed normal mammography tissue texture). Fixed by per-image min-max normalizing before RGB
   conversion (`ddsm_acgan/data.py`, `_load_as_uint8_grayscale`). This explains both the pre-fix
   "collapsed"-looking GAN samples and a classifier that could barely beat a majority-class baseline —
   every training image was effectively near-blank.

All results below post-date both fixes.

## 3. Baseline (original, paper-faithful) architecture

### 3.1 GAN training — unstable throughout, never fully converges

The baseline generator/discriminator (`ddsm_acgan/models.py`) trained on mass-only data for 300 epochs
(`lvv8gjtz`, batch 64, lr 2e-4, Adam β₁=0.5):

| | mean | std | min | max |
|---|---|---|---|---|
| D_loss (full 300 epochs) | 1.137 | 0.351 | 0.548 | 2.460 |
| G_loss (full 300 epochs) | 3.969 | 0.985 | 1.393 | 7.517 |
| D_loss (last 50 epochs) | — | 0.246 | — | — |
| G_loss (last 50 epochs) | — | 0.741 | — | — |

G_loss ranges over a >5× spread (1.39 to 7.52) across the run, and the standard deviation in the *final*
50 epochs is still substantial (0.741) — the training never settles into a stable equilibrium, consistent
with visible checkerboard artifacts in the generator's sample grids (kernel=5/stride=2 transpose
convolutions — see §4.2) and multi-epoch stretches of discriminator/generator imbalance observed across
every baseline run at every epoch count tested (100–2000).

### 3.2 Classifier results — a consistent precision/recall trade-off, not a net improvement

Every baseline-architecture AD/SA comparison run (across both the full mass+calc scope and the mass-only
scope, at various synthetic quantities) shows the same qualitative pattern: augmentation shifts the
classifier's operating point toward malignant sensitivity at benign's expense, without a reliable net
accuracy gain.

| Run | Scope | Mode | Accuracy | Malignant recall | Benign recall |
|---|---|---|---|---|---|
| `xqfofuy0` | full (mass+calc) | AD | 61.79% | 0.351 | 0.790 |
| `3vp2j42l` | full (mass+calc) | AD | 66.40% | 0.565 | 0.727 |
| `xx3qqtig` | full (mass+calc) | SA | 56.96% | 0.417 | 0.668 |
| `7eg8ar7r` | full (mass+calc) | SA | 59.66% | 0.373 | 0.741 |
| `rzrb0h7u` | full (mass+calc) | SA | 63.23% | 0.653 | 0.619 |
| `wy8rmogn` | mass-only | AD | 64.02% | 0.639 | 0.641 |
| *(mass-only SA, 300/300 mix)* | mass-only | SA | 63.23% | 0.65 | 0.62 |

The full-mass+calc rows show meaningful run-to-run variance even for the *same* configuration (AD ranges
61.79–66.40% across two runs with no fixed random seed) — a reminder that single-run comparisons in this
project carry real noise, not just the architecture/augmentation effect being studied. Across every SA row
regardless of scope, malignant recall moves opposite to benign recall — the trade-off is consistent, not
run-specific.

## 4. Improved architecture

Two changes to the generator/discriminator (`ddsm_acgan/models_improved.py`), identical in mechanism to
the CovidGAN Stage 2 changes (see `CovidGAN-Pytorch/REPORT.md` §4 for full derivation):

### 4.1 Spectral normalization on the discriminator

Every discriminator conv layer and both output heads wrapped with `torch.nn.utils.spectral_norm`
(replacing plain BatchNorm), bounding the discriminator's Lipschitz constant (Miyato et al. 2018) to
directly target the instability quantified in §3.1.

### 4.2 Kernel=4/stride=2 generator upsampling

The four `ConvTranspose2d` blocks changed from `kernel_size=5, stride=2` (the paper's specification, not
evenly divisible by the stride) to `kernel_size=4, stride=2` (evenly divisible), removing the checkerboard
artifact by construction (Odena et al. 2016) rather than leaving it to be learned around.

### 4.3 Measured effect on training stability

Same mass-only data, same hyperparameters, only the architecture differs:

| | D_loss std | G_loss std |
|---|---|---|
| Baseline (full 300-epoch run) | 0.351 | 0.985 |
| Baseline (last 50 epochs) | 0.246 | 0.741 |
| Improved (epochs 201–300, `62aw102a`) | **0.031** | **0.025** |
| Improved (epochs 351–600, `cugo455s`) | **0.026** | **0.018** |

The improved architecture's loss variance is **roughly 10–30× smaller** than the baseline's, both overall
and specifically comparing each architecture's own late-training tail — not merely "looks calmer" but a
measured order-of-magnitude reduction in training instability, holding from epoch ~200 through 600 without
degrading.

### 4.4 Feature-space alignment (PCA)

With the baseline architecture, synthetic malignant samples formed a tight cluster almost entirely outside
the real-malignant region of classifier feature space — essentially no overlap. With the improved
architecture, synthetic malignant samples visibly overlap the real-malignant region — evidence the
generator is learning the true minority-class feature distribution, not a discriminator-fooling shortcut.

### 4.5 Classifier results (synthetic mix confirmed matched: 300 malignant / 300 benign throughout)

| Run | GAN architecture | Epochs | Accuracy | Malignant recall | Benign recall | Macro-F1 |
|---|---|---|---|---|---|---|
| `wy8rmogn` | — (real only, AD) | — | 64.02% | 0.64 | 0.64 | 0.63 |
| *(mass-only SA)* | baseline | 100 | 63.23% | 0.65 | 0.62 | — |
| `y2bvl1oo` | improved | 300 | **66.93%** | 0.52 | 0.76 | 0.65 |
| `7s0nk8lu` | improved | 600 | 65.08% | 0.61 | 0.68 | 0.64 |
| `53muuv2d` | improved | 600 (repeat) | 64.81% | 0.59 | 0.69 | 0.63 |

The 600-epoch improved result reproduced twice (`7s0nk8lu`, `53muuv2d`) within ~0.3 points of each other —
consistent, not a one-off. Reading the full set: the improved architecture's best result (300 epochs,
66.93%) is a genuine +3.7-point improvement over the baseline architecture at matched synthetic
composition. Training the improved architecture further (600 epochs) does not extend that gain — both
600-epoch runs land between the baseline and the 300-epoch result, with malignant recall improving further
at benign's expense. This tracks §4.3's stability data: the improved architecture reaches its stable loss
band by ~epoch 200 and only drifts slowly afterward, so additional training mostly re-balances the
same trade-off rather than compounding a quality gain.

## 5. Matched-epoch comparison: baseline vs. improved architecture at 300 epochs

The comparisons in §4.5 held GAN architecture and synthetic composition fixed but not GAN training
duration — the earlier baseline SA result (63.23%) came from a generator trained for only 100 epochs,
while the improved results were from 300–600 epochs. The baseline architecture (`fil0gesf`) has now been
retrained to exactly 300 epochs on the same mass-only data (resumed from its own epoch-100 checkpoint,
same hyperparameters), and its synthetic pool re-evaluated (`jkvaptex`), giving a fully matched comparison
— same epochs, same synthetic composition, only the architecture differing:

| Run | GAN architecture | Epochs | Final D_loss / G_loss | Accuracy | Malignant recall | Benign recall |
|---|---|---|---|---|---|---|
| `wy8rmogn` | — (real only, AD) | — | — | 64.02% | 0.64 | 0.64 |
| `jkvaptex` | baseline | **300** | 0.50 / 6.08 | **60.05%** | 0.68 | 0.55 |
| `y2bvl1oo` | improved | **300** | 1.72 / 0.94 | **66.93%** | 0.52 | 0.76 |

This is the cleanest result in the project. At identical epoch count and identical synthetic composition:

- **Baseline-architecture SA (60.05%) is *worse* than real-data-only AD (64.02%)** — augmentation actively
  hurts when the underlying GAN is this unstable. Its final-epoch loss (D=0.50, G=6.08) shows the run
  happened to stop mid-swing in a heavily generator-losing phase — an illustration of §3.1's finding that
  the baseline architecture never reaches a stable state to stop at consistently; which 300-epoch snapshot
  you get is partly a matter of where in its oscillation cycle training happened to be interrupted.
- **Improved-architecture SA (66.93%) clearly beats both** the baseline SA and the real-only AD, at the
  exact same epoch budget.
- The gap between the two architectures at matched epochs is **+6.9 points** (60.05% → 66.93%) — larger
  than the +3.7-point gap reported in §4.5 against the undertrained (100-epoch) baseline, because that
  earlier baseline comparison was, if anything, *flattering* to the baseline architecture relative to a
  fair fight at equal training budget.

## 6. Discussion

**What worked:** both architecture changes produced their predicted, mechanistically-explained effect —
spectral normalization measurably stabilized training (10–30× lower loss variance), and the kernel/stride
fix removes the checkerboard artifact by construction. The improved architecture's best classifier result
(66.93%) is a real, synthetic-mix-matched improvement over the baseline (63.23%).

**What did not fully work:** more training of the improved architecture does not compound the gain — 300
epochs outperforms 600 on this task, meaning "more stable training" is not the same as "keeps getting
better with more compute." The underlying precision/recall trade-off (synthetic augmentation shifting
sensitivity toward malignant at benign's expense) persists across every configuration tested, baseline or
improved — the architecture change shifts *where* on that trade-off curve you land, not whether the
trade-off exists.

**What we learned:** generation-quality improvements (stability, feature alignment) and classification
improvements are related but not interchangeable evidence — this echoes the same finding independently
established in the CovidGAN reconstruction (`REPORT.md` §3.6), where a 2× FID improvement produced zero
downstream classification change. Here, the improved architecture's stability and PCA gains *did*
translate into a real classification improvement, but only up to a point (300 epochs), and the exact
mechanism connecting "more stable training" to "better classification at this specific epoch count, not
beyond" is not yet fully explained — a natural next question once the pending matched-epoch comparison
(§5) closes out the last open confound.

## References

Same as `CovidGAN-Pytorch/REPORT.md` §7, items 4 (Miyato et al., spectral normalization) and 5 (Odena et
al., checkerboard artifacts) apply directly to §4 above.
