# DDSM-ACGAN — all tests and results

*DDSM only. The COVID-CXR sub-project is deliberately excluded.*

*Every number is pulled from Weights & Biases (`maya-hayat-ariel-university/ddsm-acgan`) with the
run ID given, so anything here can be traced back to its run.*

> ⚠️ **Status:** `FINDINGS.md` and `STAGE2_FINDINGS.md` still state numbers that §6 below retracts.
> They have not been corrected yet. This document is the current state of truth.

---

## 1. The task

| | |
|---|---|
| Data | CBIS-DDSM mammography ROI patches, **mass-type only** (calcifications excluded) |
| Task | Benign vs Malignant |
| Train / Test | 1,318 / 378 (231 benign, 147 malignant) |
| Split | CBIS-DDSM's own official case-description CSVs, not re-randomized |
| Always-guess-benign floor | **61.1%** — any result must beat this to mean anything |

**The question being tested:** does adding GAN-generated fake ROIs to the training set improve the
classifier?

| Term | Meaning |
|---|---|
| **AD** | Trained on **real images only**. The baseline. |
| **SA** | Trained on **real + synthetic**. Should beat AD if augmentation works. |
| **Frozen** | VGG16 backbone locked; ~33K trainable params (the original paper's design) |
| **Unfrozen (uf=2)** | Top 2 VGG16 blocks train too; millions of params |
| **Pool** | The 600 fake images (300 benign + 300 malignant) a generator produces |
| **Pool draw** | *Which* random noise vectors made that pool — this turned out to matter a lot |

---

## 2. Two data bugs found and fixed before any result counted

| Bug | Effect | Fix |
|---|---|---|
| Mask folders globbed as images | Silently doubled image counts with binary segmentation masks | Exclude any path containing `mask(s)`/`label(s)` |
| **16-bit PNGs destroyed on load** | CBIS-DDSM ROIs are `I;16`; `PIL.convert("RGB")` truncates them to near-white — every training image was effectively blank | Per-image min-max normalise before RGB conversion |

The second one explains both the early "collapsed"-looking GAN samples and a classifier that could
barely beat the majority-class floor. **All results below post-date both fixes.**

---

## 3. The three generators being compared

| Name | Architecture | Epochs | Run ID |
|---|---|---|---|
| **baseline** | Paper-faithful (kernel 5, BatchNorm D) | 300 | `fil0gesf` |
| **improved ep300** | Spectral-norm D + kernel 4 upsampling | 300 | `62aw102a` |
| **improved ep600** | Same architecture, trained longer | 600 | `cugo455s` |

The "improved" architecture makes two changes: **spectral normalisation** on the discriminator
(bounds its Lipschitz constant, targeting training instability) and **kernel 5→4** in the generator's
upsampling (removes checkerboard artifacts by construction, since 4÷2 divides evenly).

---

## 4. Test 1 — Does the improved architecture train more stably? ✅ YES

Same data, same hyperparameters, only the architecture differs:

| | D_loss variability | G_loss variability |
|---|---|---|
| Baseline (full 300 epochs) | 0.351 | 0.985 |
| Baseline (last 50 epochs) | 0.246 | 0.741 |
| **Improved (epochs 201–300)** | **0.031** | **0.025** |
| **Improved (epochs 351–600)** | **0.026** | **0.018** |

- ✅ **10–30× lower loss variance.** Not "looks calmer" — a measured order-of-magnitude reduction.
- The baseline never settles; it swings 3–6× over 30–80 epoch windows at every epoch count tested.
- ⚠️ This is a claim about *training dynamics only*. Whether it produces better images, or a better
  classifier, are separate questions — tested in §5 and §7. **They do not follow automatically.**

---

## 5. Test 2 — Does the improved architecture make better images? ⚠️ IT DEPENDS ON WHEN YOU STOP

### 5.1 Full training curve to 900 epochs (runs `1fzjlyh6`, `dasjho3b`)

Every saved checkpoint of both architectures, one 600-image pool each, scored against the same real
reference at n=600. KID, lower = better:

| epoch | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 |
|---|---|---|---|---|---|---|---|---|---|
| **baseline** | 0.369 | 0.376 | 0.216 | 0.180 | 0.164 | 0.315 | 0.249 | 0.170 | 0.172 |
| **improved** | 0.414 | 0.327 | 0.273 | 0.403 | 0.390 | 0.339 | 0.385 | 0.283 | 0.276 |

| epoch | 500 | 550 | 600 | 650 | 700 | 750 | 800 | 850 | 900 |
|---|---|---|---|---|---|---|---|---|---|
| **baseline** | 0.158 | 0.163 | 0.134 | **0.121** ⭐ | 0.419 💥 | 0.437 | 0.435 | 0.393 | 0.404 |
| **improved** | 0.220 | 0.236 | 0.203 | 0.259 | 0.206 | **0.194** | 0.237 | 0.308 | 0.264 |

### 5.2 The baseline collapses catastrophically at ~epoch 684 and never recovers

| baseline | ep600 | ep650 | ep700 | ep750 | ep800 | ep850 | ep900 |
|---|---|---|---|---|---|---|---|
| KID | 0.134 | **0.121** | 0.419 | 0.437 | 0.435 | 0.393 | 0.404 |
| D_loss | 1.35 | 1.17 | 0.84 | 0.48 | 0.50 | 0.48 | **0.38** |
| G_loss | 1.76 | 2.32 | 5.11 | 6.87 | 7.44 | 8.76 | **7.81** |

**+245% KID in 50 epochs.** The losses show the mechanism: D_loss falls toward zero while G_loss
climbs — the discriminator overpowers the generator, G's gradients vanish, output collapses. The turn
is visible at ep684 (D 0.87→0.64, G 2.78→4.37).

The improved architecture over the same span: **D 1.60 → 1.57, G 0.90 → 0.90.** Completely flat. It
never diverges.

### 5.3 Both pre-committed criteria (agreed before the runs, to avoid cherry-picking)

| Criterion | baseline | improved | winner |
|---|---|---|---|
| **Median KID over ep650–900** | 0.4114 | **0.2482** | **improved**, by 0.163 |
| **Best-ever checkpoint** | **0.1215** (ep650) | 0.1944 (ep750) | **baseline**, by 0.073 |
| Matched-epoch record (18 pairs) | **12** | 6 | baseline wins *all* of ep150–650; improved wins *all* of ep700–900 |

Both are legitimate and answer different questions:

- **"I monitor KID and stop at the best checkpoint"** → baseline. Higher ceiling.
- **"I train to a fixed budget and take what I get"** → improved. No cliff to fall off.

### 5.4 What spectral normalisation actually bought

The same mechanism explains both halves of the trade-off:

> Spectral normalisation constrains the discriminator's Lipschitz constant — it *weakens D by
> construction*. A weaker D teaches the generator less, which costs peak quality (baseline reached
> KID 0.121; the improved architecture never got below 0.194). But a weaker D also **can never
> overpower the generator**, which is precisely the failure that destroyed the baseline at ep684.

Lower ceiling, no floor. That is the textbook spectral-norm trade-off, demonstrated end-to-end here.

⚠️ This is **one training run per architecture.** "The baseline diverges around ep700" should be
written as *this baseline run diverged at ep684*, not as a property of the architecture. A second
seed per architecture would settle it.

### 5.5 Loss stability: what it does and does not predict

An earlier draft of this document claimed a smooth loss curve "carried no information about whether
the generator was any good." The 900-epoch data **falsifies that**, and the corrected claim is
sharper:

- ❌ Loss stability does **not** predict incremental quality — the improved architecture's losses are
  10–30× smoother yet its KID swings ±50% between adjacent checkpoints.
- ✅ Loss stability **does** predict catastrophic failure. D_loss collapsing toward zero is the
  classic divergence signature and it fired exactly where KID exploded.

### 5.6 Single-checkpoint comparisons are unreliable

An early pass (`nhq3jrms`) compared baseline **ep300** against improved **ep600** and reported the
improved architecture winning by −36%, 5.5σ. The curve shows those were the baseline's *worst*
checkpoint after ep100 against the improved architecture's *best at the time*. ❌ **That −36% win is
retracted.**

| Architecture | Biggest swing between adjacent checkpoints (50 epochs apart) |
|---|---|
| baseline | **+245%** (ep650→700) |
| improved | **+49%** (ep150→200) |

Against error bars of only ±0.015. Any conclusion from one checkpoint per architecture measures where
training happened to be sampled — the same failure mode as single pool draws (§6) and single seeds.

Note that `FINDINGS.md` §5's original intuition — that the baseline ep300 checkpoint was "caught
mid-swing in a bad phase" — turns out to be **correct for generation quality**, even though §6.2
shows it is wrong for classification. Right instinct, wrong metric.

😐 In absolute terms every checkpoint is poor (well-trained GANs score FID < 50). None of these
generators produces realistic mammography.

## 6. The problem: no random seeds, and a noise floor nobody had measured

Every early result ran without fixed seeds. Once seeds were added, results changed.

### 6.1 The noise floor

| Noise source | Magnitude |
|---|---|
| Classifier random seed | up to **1.6 points** |
| **Which 600 noise vectors formed the pool** | up to **5.3 points** |
| **Regenerating a seeded pool on another GPU** (§6.4) | up to **2.65 points** |

**No effect anyone was chasing exceeded 5 points.** Results were substantially measuring randomness.

### 6.2 ❌ Retraction 1 — `FINDINGS.md` §5's "+6.9-point win"

| | baseline SA | improved SA | gap |
|---|---|---|---|
| Unseeded (the claim) | 60.05 (`jkvaptex`) | 66.93 (`y2bvl1oo`) | **+6.88** |
| Seed 0 | 65.34 (`430w9r7n`) | 65.61 (`j20fo7ro`) | +0.27 |
| Seed 1 | 65.34 (`hjozkvn0`) | 64.02 (`m95w4t67`) | −1.32 |
| **Mean** | **65.34** | **64.81** | **−0.52** |

One unlucky draw vs one lucky draw. `FINDINGS.md` §5 blamed a "bad generator checkpoint caught
mid-swing" — but the seeded runs load **that exact same checkpoint file** and score 5.3 points
higher. The checkpoint was never the variable; the noise draw was.

### 6.3 ❌ Retraction 2 — `STAGE2_FINDINGS.md` §2's "+1.33-point reproducible win"

Claimed: with the unfrozen classifier, augmentation gives a real win. Re-tested across **3
independent pool draws**, classifier seed held fixed so only the pool varies:

| Pool draw | baseline SA | improved300 SA |
|---|---|---|
| 1 | 70.63 (`5y8ju407`) | 67.46 (`tmhjvs29`) |
| 2 | 67.46 (`p60qeg51`) | 67.20 (`fte4werz`) |
| 3 | 68.78 (`4cbd3vvk`) | 64.02 (`k8jnco8o`) |
| **Mean** | **68.96** | **66.23** |
| AD (real only), same seed | **69.05** (`m53jdvq1`) | — |

- **SA (68.96) vs AD (69.05) = −0.09.** No lift at all. The 70.63 was the lucky draw.
- baseline − improved = 2.73 pts, but the *within-arm* spread is 3.2–3.4 pts.
  **The noise is bigger than the effect** (t=1.90, p ≈ 0.13). Not established.
- Direction is consistent though: baseline beats improved in **all 3** paired draws.

### 6.4 Seeded synthetic pools are not reproducible across sessions

The best-checkpoint grid (§7.3) was accidentally run three times, which exposed a third noise source.
Within a session, repeats are **bit-identical**. Across sessions, with the same `--seed`, same code
and same pool *directory name*, SA results differ:

| Run name | pass 1 | pass 2 | spread |
|---|---|---|---|
| `sa-base650-uf2-pool1` | 66.40 | 69.05 | **2.65** |
| `sa-base650-uf2-pool2` | 69.84 | 68.78 | 1.06 |
| `sa-base650-uf2-pool3` | 67.20 | 67.99 | 0.79 |
| `sa-imp750-frozen-pool1/2/3` | — | 62.70 / 64.29 / 61.64 | **0.00** (same-session repeat) |

**But AD runs — which read no pool — reproduce exactly, even across different days:**

| Config | 2026-08-22 | 2026-08-25 | Diff |
|---|---|---|---|
| AD uf=2, seed 1 | 69.05 (`m53jdvq1`) | 69.05 (`vne7xvub`) | **0.00** |
| AD frozen, seed 1 | 65.87 (`quas5r8e`) | 65.87 (`d9c1262z`) | **0.00** |

That isolates the cause. Classifier training *is* reproducible across sessions; the drift is entirely
in the synthetic pool. The runtime crashed at 18:02, `/content` was wiped, and the generation cell's
`os.path.isdir` guard found no pools and regenerated them — same seed, different GPU, **different
images**, because cuDNN picks convolution algorithms per device and the generator's forward pass
inherits that nondeterminism.

> **`generate_synthetic.py --seed N` reproduces a pool only on the machine that made it.** Every pool
> directory in this project is a one-off artifact. `torch.use_deterministic_algorithms(True)` would
> fix it, at a real performance cost.

Consequence for reporting: quote pools generated in a single session, and treat SA differences
smaller than ~2.65 points as unresolved. AD numbers do not carry this caveat.

---

## 7. Test 3 — Does augmentation help? ❌ NO

### 7.1 Unfrozen classifier (best current estimates)

| Training data | Accuracy |
|---|---|
| **Real only (AD)** | **69.44** |
| Real + baseline-GAN fakes | 68.96 |
| Real + improved ep600 fakes | 67.60 |
| Real + improved ep300 fakes | 66.23 |

### 7.2 Frozen classifier (the paper's original design)

| Training data | Seed 0 | Seed 1 | Mean |
|---|---|---|---|
| **Real only (AD)** | 66.14 | 65.87 | **66.00** |
| Real + baseline fakes | 65.34 | 65.34 | 65.34 |
| Real + improved ep300 fakes | 65.61 | 64.02 | 64.81 |
| Real + improved ep600 fakes | 65.61 | 65.61 | 65.61 |

### 7.3 The decisive test — the best generator from each architecture

`base650` (KID 0.121) is the best generator the project ever produced and had never been
classification-tested. `imp750` (KID 0.194) is the improved architecture's best. Three independent
pool draws each, classifier seed fixed at 1 (W&B group `bestckpt`, second pass):

| Arm | draw 1 | draw 2 | draw 3 | mean | sd | vs AD(s1) |
|---|---|---|---|---|---|---|
| SA `base650` (KID 0.121) | 69.05 | 68.78 | 67.99 | **68.61** | 0.55 | −0.44 |
| SA `imp750` (KID 0.194) | 67.46 | 67.99 | 67.20 | **67.55** | 0.40 | −1.50 |

**AD reference.** The SA arms all use classifier seed 1, so the matched comparator is AD at seed 1 =
**69.05** — the most reproducible number in the project, obtained identically in two independent
sessions three days apart (`m53jdvq1` 08-22, `vne7xvub` 08-25). Across classifier seeds, AD is:

| AD, uf=2 | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| | 69.84 (`un235dfj`) | 69.05 (`m53jdvq1`/`vne7xvub`) | 70.90 (`zkt37znw`) | **69.93** | 0.93 |

Against the AD *mean* the gaps widen to −1.32 (`base650`) and −2.38 (`imp750`).

Frozen classifier, same pools: AD 65.87, SA `base650` 63.93, SA `imp750` 62.87.

**The best generator in the project still does not beat real-data-only.** This was a pre-registered
prediction (recorded before the runs) and it held. The `base650` gap is inside the §6.4 noise band,
so the safe reading is *augmentation is neutral-to-negative*, not any specific magnitude.

➡️ **In both classifier configurations, adding synthetic images never beats real-only. Best case
is a tie.**

---

## 8. Test 4 — Does the classifier capacity change help? ✅ YES

| Classifier | AD accuracy (mean of 2 seeds) |
|---|---|
| Frozen (paper's design) | 66.00 |
| **Unfrozen, top 2 blocks** | **69.44** |

- ✅ **+3.44 points**, consistent at both seeds individually (66.14→69.84, 65.87→69.05).
- This is the **only** change in the project that reliably improves accuracy — and it has nothing to
  do with the GAN. Giving the classifier room to adapt its features to mammography texture helps.

---

## 9. Test 5 — The punchline: image quality vs. usefulness

### 9.1 The controlled version — same architecture, only training length differs

| Baseline generator | KID | Unfrozen accuracy |
|---|---|---|
| ep300 | 0.315 | 68.96 |
| **ep650** | **0.121** | **68.61** |

**A 2.6× better generator produced −0.35 accuracy points — i.e. nothing.**

This is the cleanest evidence in the project, because there is no architecture confound. One
variable changed (training length), image quality improved 62%, augmentation value did not move.

### 9.2 Across every pool with both measurements

| Pool | KID (lower = better) | Unfrozen accuracy |
|---|---|---|
| `base650` | 🥇 **0.121** | 68.61 |
| `imp750` | 0.194 | 67.55 |
| `imp600` | 0.203 | 67.60 |
| `base300` | 0.315 | 🥇 **68.96** |
| `imp300` | 0.339 | 66.23 |

**The best generator is not the best augmenter.** The rankings do not line up — the *worst-but-one*
pool by KID gives the highest accuracy.

This decoupling is the one finding that has never reversed across any round of re-testing, and §9.1
now supports it with a controlled single-variable comparison rather than a rank correlation.

⚠️ An earlier framing — "a 36% quality improvement produced a 1.4-point accuracy loss" — rested on
the retracted ep300-vs-ep600 quality comparison (§5.6). The decoupling itself does not depend on it.

## 10. Supporting probes

### 10.1 Can a classifier tell fake from real? — Yes, perfectly, for every pool

Held-out real-vs-synthetic discrimination at uf=2 capacity (`4rokgp51`, `6pr4el0n`, `3fogi0cj`,
`wqhkngnz`):

| Pool | Held-out accuracy | AUC |
|---|---|---|
| baseline | 100% | 1.00 |
| improved ep300 | 100% | 1.00 |
| improved ep600 | 100% | 1.00 |

Saturated — the probe can't rank them. But it does establish that **none of these pools is anywhere
near the real data distribution**, consistent with the poor absolute FID/KID values in §5.

### 10.2 Train on fakes alone, test on real

| Pool | Accuracy | Malignant recall |
|---|---|---|
| baseline (`14vh6zaj`) | 60.05% | 0.109 |
| improved ep300 (`1ppw8hwf`) | 59.79% | 0.231 |
| improved ep600 (`oy40wpcg`) | 65.08% | 0.340 |

All at or barely above the 61.1% always-guess-benign floor. The baseline pool in particular produces
a near-degenerate classifier (0.109 malignant recall). **The synthetic images carry very little
transferable signal on their own.**

---

## 11. Summary — what is actually true

### ✅ Holds up

- **Unfreezing the classifier helps: +3.44 points**, reproduces across seeds. The project's only
  reliable accuracy gain.
- **The improved GAN architecture trains far more stably** — 10–30× lower loss variance — and
  **never diverges**, holding D≈1.60/G≈0.90 flat from epoch 200 to 900.
- **The baseline architecture has a higher quality ceiling** (KID 0.121 vs 0.194) **but collapsed
  catastrophically at ~epoch 684** and never recovered (+245% KID in 50 epochs).
- **Spectral normalisation traded ceiling for reliability** — a weaker discriminator teaches less
  but can never overpower the generator. Lower peak, no cliff.
- **Loss stability predicts catastrophic failure, not incremental quality** (§5.5).
- **Augmentation does not help.** In every configuration tested — including the project's best
  generator at three independent pool draws — real-only matches or beats real+synthetic.
- **Image quality and augmentation value are decoupled** — a 2.6× KID improvement within one
  architecture changed accuracy by −0.35 points.
- **Uncontrolled sampling dominated every comparison** — ~5 points from the pool draw, ~2.65 points
  from regenerating a seeded pool on a different GPU, and 49–245% KID swings between adjacent
  checkpoints. Classifier training itself *is* reproducible; the synthetic pools are not (§6.4).

### ❌ Retracted

- `FINDINGS.md` §5's +6.9-point matched-epoch win → **−0.52 with seeds** (§6.2)
- `STAGE2_FINDINGS.md` §2's +1.33-point augmentation win → **−0.09 across 3 pool draws** (§6.3)
- The improved architecture's **−36% image-quality win** → worst-vs-best checkpoint selection (§5.6)
- The claim that loss stability "carries no information about generator quality" → it predicts
  divergence, just not incremental quality (§5.5)
- An earlier reading of §6.4 that blamed cross-session *classifier* nondeterminism → AD runs
  reproduce exactly across sessions; the drift is entirely in seeded pool generation
- `ANALYSIS.md` §4.2's "manifold overlap" mechanism — failed three separate tests

### The Stage 2a scorecard

| Claim | Status |
|---|---|
| Improved architecture trains more stably (loss variance) | ✅ True |
| Improved architecture is divergence-proof | ✅ **True** — flat to ep900 while the baseline collapsed |
| Improved architecture is more reliable at a fixed budget | ✅ **True** — median KID 0.248 vs 0.411 over ep650–900 |
| Improved architecture has a higher quality ceiling | ❌ False — baseline peaks better (0.121 vs 0.194) |
| Improved architecture helps classification | ❌ False |

The honest Stage 2a conclusion: **spectral normalisation traded peak generation quality for immunity
to divergence, and neither property translated into classification gains.**

## 12. One-sentence version

> On CBIS-DDSM, GAN augmentation does not improve benign/malignant classification in any
> configuration tested, and the Stage 2 architecture change stabilised the loss curve without
> improving either image quality or accuracy — the only reliable gain came from giving the
> classifier more capacity, and every other reported gain, including our own, turned out to be a
> single sample from a high-variance process that nobody had measured.

---

## 13. Outstanding corrections

| Document | Section | Action |
|---|---|---|
| `FINDINGS.md` | §5 | Retract; replace with §6.2 above |
| `FINDINGS.md` | §4.5 | Downgrade — shares the uncontrolled pool draw |
| `FINDINGS.md` | §6 | Remove "strongest, most controlled result" framing |
| `STAGE2_FINDINGS.md` | §2 | Retract finding #2 per §6.3 |
| `STAGE2_FINDINGS.md` | §3 | Note the interaction hypothesis is now unsupported |
| `ANALYSIS.md` | §4.2 | Retire the mechanism |

---

## Appendix — every W&B run

*All 90 runs in `maya-hayat-ariel-university/ddsm-acgan`, complete. Crashed and duplicate runs are
included deliberately: several conclusions in this document depend on knowing which runs were
superseded, and §6.4's cross-session finding is only visible because duplicates exist.*

*Reading note: `frozen` = the paper's original classifier (~33K trainable params); `uf=2` = top two
VGG16 blocks unfrozen. Test accuracy is on the 378-image real held-out split; the
always-guess-benign floor is 61.11%.*

### A. GAN training runs

| Run ID | Name | Arch | Epochs | Resumed from | Final D_loss | Final G_loss | State | Date |
|---|---|---|---|---|---|---|---|---|
| `o7w8kn26` | gan | baseline | 100 | — | 1.750 | 3.381 | finished | 2026-08-17 |
| `l60hw8i6` | gan | baseline | 200 | ep0100 | 1.503 | 1.598 | finished | 2026-08-17 |
| `d6b7y3il` | gan | baseline | 300 | ep0200 | 1.235 | 2.291 | finished | 2026-08-17 |
| `upk6o601` | gan | baseline | 500 | ep0300 | 1.058 | 2.796 | finished | 2026-08-18 |
| `lvv8gjtz` | gan_mass | baseline | 300 | — | 1.167 | 2.569 | finished | 2026-08-18 |
| `6wf75veh` | gan | baseline | 1000 | — | 1.346 | 3.826 | finished | 2026-08-19 |
| `dtuycho9` | gan | baseline | 1000 | ep0050 | 1.135 | 3.941 | crashed | 2026-08-19 |
| `stl77do7` | gan_improved_2_mass_only(1) | improved | 300 | — | 1.724 | 0.999 | crashed | 2026-08-19 |
| `62aw102a` | gan_improved_2_mass_only(2) | improved | 300 | ep0200 | 1.718 | 0.941 | finished | 2026-08-19 |
| `zl9op8ry` | gan_improved_2_mass_only(3) | improved | 600 | ep0300 | 1.659 | 0.938 | crashed | 2026-08-19 |
| `cugo455s` | gan_improved_2_mass_only(4) | improved | 600 | ep0350 | 1.624 | 0.916 | finished | 2026-08-21 |
| `0ju0yqg3` | gan_mass_baseline_to900 | baseline | 900 | final | 0.381 | 7.807 | finished | 2026-08-24 |
| `mhg863gg` | gan_mass_improved_to900 | improved | 900 | ep0600 | 1.573 | 0.899 | finished | 2026-08-24 |

### B. Classifier runs — AD and SA

| Run ID | Name | Mode | Classifier | Seed | Synthetic pool | Train acc | **Test acc %** | Mal recall | Ben recall | State | Date |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `7ksbu2k3` | cnn-ad | AD | frozen | — | — | 0.8233 | **59.80** | 0.446 | 0.696 | finished | 2026-08-17 |
| `jkwd8l3w` | cnn-sa | SA | frozen | — | synthetic | 0.8834 | **59.09** | 0.362 | 0.738 | finished | 2026-08-17 |
| `inhlbada` | cnn-sa | SA | frozen | — | synthetic | — | **—** | — | — | finished | 2026-08-18 |
| `xx3qqtig` | cnn-sa | SA | frozen | — | synthetic | 0.7820 | **56.96** | 0.417 | 0.668 | finished | 2026-08-18 |
| `xqfofuy0` | cnn-ad | AD | frozen | — | — | 0.7465 | **61.79** | 0.351 | 0.790 | finished | 2026-08-18 |
| `7eg8ar7r` | cnn-sa | SA | frozen | — | synthetic | 0.7608 | **59.66** | 0.373 | 0.741 | finished | 2026-08-18 |
| `1ikkjz5y` | cnn-ad | AD | frozen | — | — | 0.7170 | **—** | — | — | killed | 2026-08-18 |
| `zl072ohj` | cnn-sa | SA | frozen | — | synthetic | — | **—** | — | — | crashed | 2026-08-18 |
| `rzrb0h7u` | cnn-sa | SA | frozen | — | synthetic | 0.8288 | **63.23** | 0.653 | 0.619 | finished | 2026-08-18 |
| `3vp2j42l` | cnn-ad | AD | frozen | — | — | 0.7595 | **66.40** | 0.565 | 0.727 | finished | 2026-08-18 |
| `rt18z7v1` | cnn-sa | SA | frozen | — | synthetic | — | **—** | — | — | finished | 2026-08-18 |
| `pj1i7a2t` | cnn-sa | SA | frozen | — | synthetic | — | **—** | — | — | finished | 2026-08-18 |
| `y2bvl1oo` | cnn-sa-mass-part2 | SA | frozen | — | synthetic_improved | 0.8279 | **66.93** | 0.524 | 0.762 | finished | 2026-08-19 |
| `wy8rmogn` | cnn-ad-mass-part2 | AD | frozen | — | — | 0.7754 | **64.02** | 0.639 | 0.641 | finished | 2026-08-19 |
| `7s0nk8lu` | cnn-sa-mass-part2 | SA | frozen | — | synthetic_improved | 0.8274 | **65.08** | 0.605 | 0.680 | finished | 2026-08-21 |
| `53muuv2d` | cnn-sa-mass-part2 | SA | frozen | — | synthetic_improved | 0.8191 | **64.81** | 0.585 | 0.688 | finished | 2026-08-21 |
| `jkvaptex` | cnn-sa-mass-part1 | SA | frozen | — | synthetic | 0.8660 | **60.05** | 0.680 | 0.550 | finished | 2026-08-21 |
| `5awve1rc` | cnn-ad-seed_0 | AD | frozen | 0 | — | 0.7982 | **66.14** | 0.558 | 0.727 | finished | 2026-08-22 |
| `430w9r7n` | cnn-sa-seed_0 | SA | frozen | 0 | synthetic | 0.8712 | **65.34** | 0.551 | 0.719 | finished | 2026-08-22 |
| `j20fo7ro` | cnn-sa-improved-gan-seed_0 | SA | frozen | 0 | synthetic_improved | 0.8191 | **65.61** | 0.599 | 0.693 | finished | 2026-08-22 |
| `un235dfj` | cnn-ad-uf2-seed_0 | AD | uf=2 | 0 | — | 0.9833 | **69.84** | 0.558 | 0.788 | finished | 2026-08-22 |
| `qmz1jili` | cnn-sa-uf2-seed_0 | SA | uf=2 | 0 | synthetic | 0.9943 | **70.90** | 0.707 | 0.710 | finished | 2026-08-22 |
| `kse58jjf` | cnn-sa-improved-gan-uf2-seed_0 | SA | uf=2 | 0 | synthetic_improved | 0.9911 | **65.87** | 0.667 | 0.654 | finished | 2026-08-22 |
| `quas5r8e` | cnn-ad-seed_1 | AD | frozen | 1 | — | 0.7898 | **65.87** | 0.490 | 0.766 | finished | 2026-08-22 |
| `hjozkvn0` | cnn-sa-seed_1 | SA | frozen | 1 | synthetic | 0.8624 | **65.34** | 0.578 | 0.701 | finished | 2026-08-22 |
| `m95w4t67` | cnn-sa-improved-gan-seed_1 | SA | frozen | 1 | synthetic_improved | 0.8243 | **64.02** | 0.585 | 0.675 | finished | 2026-08-22 |
| `m53jdvq1` | cnn-ad-uf2-seed_1 | AD | uf=2 | 1 | — | 0.9947 | **69.05** | 0.687 | 0.693 | finished | 2026-08-22 |
| `5y8ju407` | cnn-sa-uf2-seed_1 | SA | uf=2 | 1 | synthetic | 0.9974 | **70.63** | 0.660 | 0.736 | finished | 2026-08-22 |
| `tmhjvs29` | cnn-sa-improved-gan-uf2-seed_1 | SA | uf=2 | 1 | synthetic_improved | 0.9922 | **67.46** | 0.653 | 0.688 | finished | 2026-08-22 |
| `u0vf4ipt` | cnn-sa-improved-gan600-seed_1 | SA | frozen | 1 | synthetic_improved_epoch600 | 0.8332 | **65.61** | 0.524 | 0.740 | finished | 2026-08-22 |
| `7yy2seyj` | cnn-sa-improved-gan-uf2-seed_1 | SA | uf=2 | 1 | synthetic_improved_epoch600 | 0.9932 | **68.52** | 0.667 | 0.697 | finished | 2026-08-22 |
| `p60qeg13` | poolvar-base-pool2 | SA | uf=2 | 1 | pool_base_s2 | 0.9969 | **67.46** | 0.762 | 0.619 | finished | 2026-08-22 |
| `4cbd3vvk` | poolvar-base-pool3 | SA | uf=2 | 1 | pool_base_s3 | 0.9927 | **68.78** | 0.680 | 0.693 | finished | 2026-08-22 |
| `fte4werz` | poolvar-imp300-pool2 | SA | uf=2 | 1 | pool_imp300_s2 | 0.9958 | **67.20** | 0.619 | 0.706 | finished | 2026-08-22 |
| `k8jnco8o` | poolvar-imp300-pool3 | SA | uf=2 | 1 | pool_imp300_s3 | 0.9927 | **64.02** | 0.667 | 0.623 | finished | 2026-08-22 |
| `k94eo6m9` | cnn-sa-improved-gan600-uf2-seed_0 | SA | uf=2 | 0 | synthetic_improved_epoch600 | 0.9953 | **66.67** | 0.728 | 0.628 | finished | 2026-08-22 |
| `vto2hoqs` | cnn-sa-improved-gan600-seed_0 | SA | frozen | 0 | synthetic_improved_epoch600 | 0.8233 | **65.61** | 0.537 | 0.732 | finished | 2026-08-22 |
| `d9c1262z` | ad-frozen-seed1 | AD | frozen | 1 | — | 0.7898 | **65.87** | 0.490 | 0.766 | finished | 2026-08-25 |
| `vne7xvub` | ad-uf2-seed1 | AD | uf=2 | 1 | — | 0.9947 | **69.05** | 0.687 | 0.693 | finished | 2026-08-25 |
| `mt2mfu16` | sa-base650-frozen-pool1 | SA | frozen | 1 | best_base650_p1 | 0.8436 | **63.49** | 0.558 | 0.684 | finished | 2026-08-25 |
| `dfrbpss7` | sa-base650-frozen-pool2 | SA | frozen | 1 | best_base650_p2 | 0.8551 | **64.55** | 0.565 | 0.697 | finished | 2026-08-25 |
| `9khoit6s` | sa-base650-frozen-pool3 | SA | frozen | 1 | best_base650_p3 | 0.8509 | **62.96** | 0.531 | 0.693 | finished | 2026-08-25 |
| `7oemgubw` | sa-base650-uf2-pool1 | SA | uf=2 | 1 | best_base650_p1 | 0.9911 | **66.40** | 0.721 | 0.628 | finished | 2026-08-25 |
| `r9drbsxn` | sa-base650-uf2-pool2 | SA | uf=2 | 1 | best_base650_p2 | 0.9974 | **69.84** | 0.714 | 0.688 | finished | 2026-08-25 |
| `nn458l72` | sa-base650-uf2-pool3 | SA | uf=2 | 1 | best_base650_p3 | 0.9990 | **67.20** | 0.673 | 0.671 | finished | 2026-08-25 |
| `v3x8xo2r` | sa-imp750-frozen-pool1 | SA | frozen | 1 | best_imp750_p1 | 0.8290 | **62.17** | 0.619 | 0.623 | finished | 2026-08-25 |
| `7ugebtjl` | sa-imp750-frozen-pool2 | SA | frozen | 1 | best_imp750_p2 | 0.7680 | **—** | — | — | crashed | 2026-08-25 |
| `1wnr289v` | ad-frozen-seed2 | AD | frozen | 2 | — | 0.7967 | **66.14** | 0.558 | 0.727 | finished | 2026-08-25 |
| `zkt37znw` | ad-uf2-seed2 | AD | uf=2 | 2 | — | 1.0000 | **70.90** | 0.694 | 0.719 | finished | 2026-08-25 |
| `hi1nckfl` | sa-base650-frozen-pool1 | SA | frozen | 1 | best_base650_p1 | 0.8420 | **64.29** | 0.578 | 0.684 | finished | 2026-08-25 |
| `xkwhrvj7` | sa-base650-frozen-pool2 | SA | frozen | 1 | best_base650_p2 | 0.8535 | **64.02** | 0.571 | 0.684 | finished | 2026-08-25 |
| `kx1x41u5` | sa-base650-frozen-pool3 | SA | frozen | 1 | best_base650_p3 | 0.8457 | **63.49** | 0.531 | 0.701 | finished | 2026-08-25 |
| `2gzdjtvj` | sa-base650-uf2-pool1 | SA | uf=2 | 1 | best_base650_p1 | 0.9969 | **69.05** | 0.707 | 0.680 | finished | 2026-08-25 |
| `fx8ijyvc` | sa-base650-uf2-pool2 | SA | uf=2 | 1 | best_base650_p2 | 0.9974 | **68.78** | 0.701 | 0.680 | finished | 2026-08-25 |
| `mirxvw6q` | sa-base650-uf2-pool3 | SA | uf=2 | 1 | best_base650_p3 | 0.9995 | **67.99** | 0.721 | 0.654 | finished | 2026-08-25 |
| `ytpur0z5` | sa-imp750-frozen-pool1 | SA | frozen | 1 | best_imp750_p1 | 0.8227 | **62.70** | 0.646 | 0.615 | finished | 2026-08-25 |
| `x5dq1v8h` | sa-imp750-frozen-pool2 | SA | frozen | 1 | best_imp750_p2 | 0.8149 | **64.29** | 0.605 | 0.667 | finished | 2026-08-25 |
| `zbbxp5ea` | sa-imp750-frozen-pool3 | SA | frozen | 1 | best_imp750_p3 | 0.8123 | **61.64** | 0.571 | 0.645 | finished | 2026-08-25 |
| `15aab19e` | sa-imp750-uf2-pool1 | SA | uf=2 | 1 | best_imp750_p1 | 0.9953 | **67.46** | 0.714 | 0.649 | finished | 2026-08-25 |
| `wz3t79no` | sa-imp750-uf2-pool2 | SA | uf=2 | 1 | best_imp750_p2 | 0.9995 | **67.99** | 0.694 | 0.671 | finished | 2026-08-25 |
| `yy0zfud8` | sa-imp750-uf2-pool3 | SA | uf=2 | 1 | best_imp750_p3 | 0.9974 | **67.20** | 0.728 | 0.636 | finished | 2026-08-25 |
| `sjoszmbl` | sa-imp750-frozen-pool1 | SA | frozen | 1 | best_imp750_p1 | 0.8227 | **62.70** | 0.646 | 0.615 | finished | 2026-08-25 |
| `z8zpmnp1` | sa-imp750-frozen-pool2 | SA | frozen | 1 | best_imp750_p2 | 0.8149 | **64.29** | 0.605 | 0.667 | finished | 2026-08-25 |
| `npn3qbco` | sa-imp750-frozen-pool3 | SA | frozen | 1 | best_imp750_p3 | 0.8123 | **61.64** | 0.571 | 0.645 | finished | 2026-08-25 |
| `pul7pghm` | sa-imp750-uf2-pool1 | SA | uf=2 | 1 | best_imp750_p1 | 0.9953 | **67.46** | 0.714 | 0.649 | finished | 2026-08-25 |
| `ep0r3gfo` | sa-imp750-uf2-pool2 | SA | uf=2 | 1 | best_imp750_p2 | 0.9995 | **67.99** | 0.694 | 0.671 | running | 2026-08-25 |

### C. Synthetic-only probes (trained on fakes alone)

| Run ID | Name | Pool | Classifier | Seed | Train acc | Test acc % | Mal recall | Ben recall |
|---|---|---|---|---|---|---|---|---|
| `14vh6zaj` | probe-baseline-synth-only | synthetic | uf=2 | 1 | 1.0000 | 60.05 | 0.109 | 0.913 |
| `1ppw8hwf` | probe-improved-synth-only-epoch300 | synthetic_improved | uf=2 | 1 | 0.9883 | 59.79 | 0.231 | 0.831 |
| `oy40wpcg` | probe-improved-synth-only-epoch600 | synthetic_improved_epoch600 | uf=2 | 1 | 0.9967 | 65.08 | 0.340 | 0.848 |

### D. Real-vs-synthetic discriminability probes

| Run ID | Name | Pool | Seed | Held-out acc % | AUC |
|---|---|---|---|---|---|
| `4rokgp51` | realness-baseline-seed0 | synthetic | 0 | 100.00 | 1.00 |
| `6pr4el0n` | realness-improved300-seed0 | synthetic_improved | 0 | 100.00 | 1.00 |
| `3fogi0cj` | realness-improved600-seed0 | synthetic_improved_epoch600 | 0 | 100.00 | 1.00 |
| `wqhkngnz` | realness-baseline-seed1 | synthetic | 1 | 100.00 | 1.00 |

### E. FID / KID evaluation runs


**`xjvjc7eo` — fid-kid-mass-part1-vs-improved2** (2026-08-24, n=600 per pool)

| Pool | FID | KID | ± |
|---|---|---|---|
| improved2 | 348.60 | 0.41248 | 0.00000 |
| part1 | 322.05 | 0.34956 | 0.00000 |

**`nhq3jrms` — fid-kid-mass-pools** (2026-08-24, n=600 per pool)

| Pool | FID | KID | ± |
|---|---|---|---|
| baseline | 312.10 | 0.31918 | 0.01292 |
| improved300 | 301.66 | 0.33825 | 0.01441 |
| improved600 | 210.61 | 0.20479 | 0.01630 |

**`1fzjlyh6` — fid-kid-training-curve** (2026-08-24, n=600 per pool)

| Pool | FID | KID | ± |
|---|---|---|---|
| base0050 | 340.32 | 0.36974 | 0.01500 |
| base0100 | 337.59 | 0.37745 | 0.01284 |
| base0150 | 236.26 | 0.21483 | 0.01370 |
| base0200 | 193.38 | 0.18409 | 0.01392 |
| base0250 | 198.59 | 0.16532 | 0.01717 |
| base0300 | 312.10 | 0.31702 | 0.01132 |
| imp0050 | 356.60 | 0.41178 | 0.01487 |
| imp0100 | 295.31 | 0.32512 | 0.01427 |
| imp0150 | 252.88 | 0.27283 | 0.01347 |
| imp0200 | 335.26 | 0.40689 | 0.01720 |
| imp0250 | 338.27 | 0.38996 | 0.01487 |
| imp0300 | 301.66 | 0.34143 | 0.01463 |
| imp0350 | 333.09 | 0.38481 | 0.01594 |
| imp0400 | 262.19 | 0.28464 | 0.01470 |
| imp0450 | 258.07 | 0.27606 | 0.01296 |
| imp0500 | 215.00 | 0.21828 | 0.01250 |
| imp0550 | 237.73 | 0.23547 | 0.01474 |
| imp0600 | 210.61 | 0.20089 | 0.01440 |

**`dasjho3b` — fid-kid-curve-900** (2026-08-25, n=600 per pool)

| Pool | FID | KID | ± |
|---|---|---|---|
| base0050 | 340.32 | 0.36948 | 0.01409 |
| base0100 | 337.59 | 0.37580 | 0.01331 |
| base0150 | 236.27 | 0.21550 | 0.01554 |
| base0200 | 193.38 | 0.17983 | 0.01595 |
| base0250 | 198.60 | 0.16377 | 0.01458 |
| base0300 | 312.10 | 0.31492 | 0.01154 |
| base0350 | 246.16 | 0.24883 | 0.01850 |
| base0400 | 197.47 | 0.16982 | 0.01315 |
| base0450 | 202.24 | 0.17181 | 0.01390 |
| base0500 | 184.32 | 0.15830 | 0.01519 |
| base0550 | 190.33 | 0.16300 | 0.01418 |
| base0600 | 168.12 | 0.13426 | 0.01300 |
| base0650 | 147.23 | 0.12146 | 0.01493 |
| base0700 | 369.76 | 0.41861 | 0.01708 |
| base0750 | 383.04 | 0.43675 | 0.01823 |
| base0800 | 384.51 | 0.43512 | 0.02141 |
| base0850 | 343.66 | 0.39346 | 0.01490 |
| base0900 | 372.31 | 0.40428 | 0.01667 |
| imp0050 | 356.60 | 0.41382 | 0.01924 |
| imp0100 | 295.30 | 0.32743 | 0.01228 |
| imp0150 | 252.88 | 0.27315 | 0.01096 |
| imp0200 | 335.26 | 0.40306 | 0.01552 |
| imp0250 | 338.27 | 0.38977 | 0.01583 |
| imp0300 | 301.66 | 0.33864 | 0.01614 |
| imp0350 | 333.09 | 0.38500 | 0.01675 |
| imp0400 | 262.19 | 0.28321 | 0.01478 |
| imp0450 | 258.07 | 0.27552 | 0.01216 |
| imp0500 | 215.00 | 0.22005 | 0.01182 |
| imp0550 | 237.73 | 0.23602 | 0.01218 |
| imp0600 | 210.61 | 0.20268 | 0.01711 |
| imp0650 | 251.28 | 0.25923 | 0.01459 |
| imp0700 | 203.92 | 0.20616 | 0.01516 |
| imp0750 | 201.76 | 0.19440 | 0.01417 |
| imp0800 | 233.47 | 0.23724 | 0.01748 |
| imp0850 | 267.44 | 0.30787 | 0.02038 |
| imp0900 | 249.47 | 0.26426 | 0.01448 |

---

*Regenerate this appendix:*

```python
import wandb
for r in sorted(wandb.Api().runs("maya-hayat-ariel-university/ddsm-acgan"),
                key=lambda r: r.created_at):
    print(r.id, r.name, r.state, r.config.get("seed"), r.summary.get("test_accuracy"))
```
