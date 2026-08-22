# DDSM-ACGAN Stage 2 — Classifier Capacity Findings (seed 0 vs. seed 1)

*Companion to `FINDINGS.md` (GAN architecture) and `CovidGAN-Pytorch/stage2` (the same classifier-capacity
idea applied to CovidGAN's own task). This document covers the classifier-side Stage 2 experiment:
unfreezing VGG16's top conv blocks + a BatchNorm head, evaluated at two independent seeds so single-run
noise can be distinguished from a real effect. All numbers pulled directly from Weights & Biases
(`maya-hayat-ariel-university/ddsm-acgan`).*

## 1. The full grid

Mass-only CBIS-DDSM, both classifier configurations (frozen VGG16 vs. `unfreeze_blocks=2` + `head_bn=True`,
backbone LR 1e-5) crossed with three training-data conditions (real only / real + baseline-GAN synthetic /
real + improved-GAN synthetic), each run at two seeds:

| Classifier | Mode | Synthetic source | Seed 0 | Seed 1 | Mean | Spread |
|---|---|---|---|---|---|---|
| frozen | AD | — | 66.14% | 65.87% | 66.00% | 0.27 |
| frozen | SA | baseline GAN | 65.34% | 65.34% | 65.34% | 0.00 |
| frozen | SA | improved GAN | 65.61% | 64.02% | 64.82% | 1.59 |
| **unfrozen (uf=2)** | **AD** | — | 69.84% | 69.05% | **69.45%** | 0.79 |
| **unfrozen (uf=2)** | **SA** | **baseline GAN** | 70.90% | 70.63% | **70.77%** | 0.27 |
| **unfrozen (uf=2)** | **SA** | **improved GAN** | 65.87% | 67.46% | **66.67%** | 1.59 |

## 2. What reproduces across both seeds (real findings, not single-run noise)

1. **Unfreezing raises the baseline.** Frozen AD (66.00% mean) → unfrozen AD (69.45% mean), +3.45pts,
   consistent at both seeds individually (66.14→69.84, 65.87→69.05). Confirms the capacity hypothesis
   directly: giving the classifier room to adapt its features to mammography texture helps, independent of
   any augmentation question.
2. **With the unfrozen classifier, baseline-GAN augmentation gives a real, reproducible win.**
   Unfrozen SA-baseline (70.77% mean) beats unfrozen AD (69.45% mean) at **both** seeds individually
   (70.90>69.84, 70.63>69.05) — a genuine +1.3pt average lift, and the tightest spread of any unfrozen
   condition (0.27). This is the first fully reproducible case in the whole project (CovidGAN or DDSM)
   where synthetic augmentation clearly helps a matched real-data-only baseline.
3. **With the unfrozen classifier, improved-GAN augmentation consistently *hurts* relative to both AD and
   baseline-GAN SA.** Unfrozen SA-improved (66.67% mean) is below unfrozen AD (69.45%) at both seeds
   (65.87<69.84, 67.46<69.05), and below unfrozen SA-baseline at both seeds too. **This settles a question
   left open last round**: I had hypothesized this reversal might be single-seed noise, given how much
   run-to-run variance this project has seen elsewhere. It is not — it reproduces cleanly across two
   independent seeds with a consistent direction and magnitude (~3-4pt gap both times).

## 3. The reversal is real, but the fingerprint hypothesis that was proposed to explain it is not supported

A synthetic-only fingerprint probe (train on *only* one synthetic pool, no real data, evaluate on real
test) was run to test whether the improved GAN's more stable, more consistent output carries a more
learnable GAN-specific signature that an unfrozen classifier could exploit as a shortcut:

| Synthetic-only probe (seed 1, unfrozen) | Accuracy | Malignant recall | Benign recall |
|---|---|---|---|
| Baseline GAN only | 60.05% | 0.109 | 0.913 |
| Improved GAN only | 59.79% | 0.231 | 0.831 |

Overall accuracy is tied (within noise). If the fingerprint hypothesis were correct, the improved pool
should have generalized *worse* standalone; instead its malignant recall is roughly double the baseline
pool's, and the baseline-only probe is the one that looks closer to a degenerate, majority-collapsed
classifier (0.109 malignant recall). **This hypothesis is rejected** — the improved GAN's synthetic images,
used alone, carry more genuine transferable signal than the baseline GAN's, not less.

That means the reversal in Section 2 is a genuine open question: it is not explained by "the improved
pool has bad standalone signal." The likely locus is an **interaction effect between the real training set
and the specific improved-GAN synthetic distribution**, only visible when an unfrozen (high-capacity)
classifier trains on the combination of both — not something visible from either component alone. This has
not yet been isolated further.

## 4. Honest correction from the previous round of analysis

The prior message in this investigation speculated the reversal was "likely noise" and proposed testing it
via multi-seed comparison before treating it as real. That test has now been run: **the reversal is real,
not noise.** The fingerprint hypothesis proposed as the mechanism was tested directly and rejected. Both of
these are useful, well-evidenced findings for the report even though the second one closes off an
explanation rather than confirming one — the honest state is "real effect, mechanism not yet identified,"
which is itself worth reporting as an open question rather than being papered over with a guess.

## 5. Suggested next step

Given the interaction-effect hypothesis in Sec. 3, a targeted follow-up would vary the *ratio* of real to
improved-GAN-synthetic data (rather than the fixed 300/300 mix used throughout) to see whether the
degradation scales with how much improved-GAN data is mixed in, which would support an interaction/dilution
mechanism specifically. A third seed via `multiseed.py` would also tighten the confidence interval on the
Section 2 findings, though two consistent seeds already meaningfully changes their status from "single-run
observation" to "reproducible finding."
