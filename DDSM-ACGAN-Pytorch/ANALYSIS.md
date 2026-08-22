# Analysis: why the improved GAN "won" unseeded and lost once seeds were fixed

*A re-audit of every run in `maya-hayat-ariel-university/ddsm-acgan` (44 runs, pulled via the W&B
public API, not from console memory), prompted by an apparent contradiction: the improved GAN
architecture beat the baseline architecture decisively in the unseeded runs (`FINDINGS.md` §5) but
lost to it in the seeded Stage 2 grid (`STAGE2_FINDINGS.md` §1–2).*

*The audit finds these are **two separate phenomena**, not one. The first is a measurement artifact
and retracts a headline result. The second is probably a real effect, and this document proposes a
mechanism for it that fits all the available evidence — including the observations that caused the
previously-proposed mechanism to be rejected.*

---

## 1. Summary of conclusions

1. **`FINDINGS.md` §5's "+6.9-point matched-epoch win" does not survive seeding and should be
   retracted.** Re-running the identical configuration with fixed seeds gives −0.52 points (improved
   marginally *worse*), not +6.9.
2. **The explanation §5 gave for that result is refuted by the data.** §5 blamed a bad baseline-GAN
   checkpoint caught "mid-swing"; the seeded runs load that same checkpoint file and score 5.3 points
   higher. The variable was never the checkpoint.
3. **The largest noise source in this project is the synthetic pool's z-draw (~5 accuracy points) —
   larger than any effect either findings document reports.** It was uncontrolled in every run before
   commit `98e2887`.
4. **`STAGE2_FINDINGS.md` §2's "reproduces across two independent seeds" overstates the evidence.**
   Both seeds share a single synthetic pool draw per arm, so the seeds vary the ~1.6-point noise term
   and hold the ~5-point one fixed.
5. **The unfrozen-classifier reversal is nonetheless likely real**, and §4 below proposes a mechanism —
   *manifold overlap removes the classifier's ability to quarantine synthetic data* — that explains it,
   explains why it appears only when the backbone is unfrozen, and reconciles the synthetic-only probe
   result that killed the earlier fingerprint hypothesis.

---

## 2. The retraction: `FINDINGS.md` §5 is a lucky-draw/unlucky-draw artifact

All rows below are the frozen classifier (`unfreeze_blocks=0`), mass-only CBIS-DDSM, 25 epochs,
300/300 synthetic composition, evaluated on the same 378-image real test split.

| Condition | Unseeded | seed 0 | seed 1 | seed mean |
|---|---|---|---|---|
| AD (real only) | 64.02 (`wy8rmogn`) | 66.14 (`5awve1rc`) | 65.87 (`quas5r8e`) | 66.00 |
| SA, baseline GAN (ep300) | **60.05** (`jkvaptex`) | 65.34 (`430w9r7n`) | 65.34 (`hjozkvn0`) | **65.34** |
| SA, improved GAN (ep300) | **66.93** (`y2bvl1oo`) | 65.61 (`j20fo7ro`) | 64.02 (`m95w4t67`) | **64.81** |

`FINDINGS.md` §5 — described there as "the cleanest result in the project" — is the comparison of the
two bolded unseeded cells: 60.05 vs 66.93, +6.88 points for the improved architecture. Re-run with
seeds:

| | baseline-GAN SA | improved-GAN SA | difference |
|---|---|---|---|
| unseeded (the §5 claim) | 60.05 | 66.93 | **+6.88** |
| seed 0 | 65.34 | 65.61 | +0.27 |
| seed 1 | 65.34 | 64.02 | −1.32 |
| **seed mean** | **65.34** | **64.81** | **−0.52** |

The effect does not merely shrink — it reverses sign and lands inside the noise band. `FINDINGS.md`
§5's three bullet conclusions all fail with it:

- "Baseline-architecture SA (60.05%) is *worse* than real-data-only AD (64.02%) — augmentation
  actively hurts when the underlying GAN is this unstable." → At fixed seeds, baseline SA is 65.34
  and AD is 66.00. The gap is 0.7 points, not 4, and the "augmentation actively hurts" reading does
  not hold.
- "Improved-architecture SA clearly beats both." → It beats neither at the seed mean.
- "The gap at matched epochs is +6.9 points… larger than the +3.7 reported in §4.5." → Both gaps are
  measurements of the same uncontrolled noise term.

### 2.1 The §5 mechanism claim is directly refuted, not merely unsupported

`FINDINGS.md` §5 attributes `jkvaptex`'s 60.05% to the generator checkpoint: *"Its final-epoch loss
(D=0.50, G=6.08) shows the run happened to stop mid-swing in a heavily generator-losing phase… which
300-epoch snapshot you get is partly a matter of where in its oscillation cycle training happened to
be interrupted."*

That is testable, and the seeded runs test it. Both `jkvaptex` and the two seeded baseline-SA runs
sample **the same file** — `fixed_gan_mass/checkpoints/ddsm_acgan_final.pt`, written by `fil0gesf` at
exactly the D=0.496 / G=6.084 state the passage describes (notebook cell 20 in
`DDSM_Colab-seed1run.ipynb`; `train_gan.py`'s checkpoints are named by absolute epoch and `--epochs`
is a total, so no later run overwrote it). Identical generator weights, identical epoch count.

Result: 60.05% unseeded, 65.34% at both seeds. **The checkpoint was never the variable.** The only
thing that changed is which 600 noise vectors `generate_synthetic.py` happened to draw.

---

## 3. Variance decomposition: the z-draw dominates everything

Separating the two randomness sources that the seeded runs let us isolate:

| Source | How it is measured here | Magnitude |
|---|---|---|
| Classifier training only (init, shuffling) | seed 0 vs seed 1, **same** synthetic pool | 0.00 – **1.59** pts |
| Synthetic pool z-draw + classifier training | unseeded vs seed-mean, same GAN checkpoint | up to **5.29** pts |

Per-condition classifier-seed spread: 0.27 (frozen AD), 0.00 (frozen SA-baseline), 1.59 (frozen
SA-improved), 0.79 (unfrozen AD), 0.27 (unfrozen SA-baseline), 1.59 (unfrozen SA-improved).

Since classifier-seed variance is bounded at ~1.6 points, the 5.29-point baseline-SA swing cannot be
attributed to classifier training. **Roughly 3.5–5 points of it come from the pool draw alone.**

This is the headline methodological finding of the audit: *which 600 z-vectors you sample* is a larger
effect than any architecture, augmentation, or capacity difference this project has reported. Every
result predating `98e2887` ("Add --seed to generate_synthetic.py") carries this term uncontrolled,
including `FINDINGS.md` §3.2, §4.5 and §5.

It also retroactively explains the run-to-run variance flagged but unexplained in `FINDINGS.md` §3.2
(AD ranging 61.79–66.40% across two identical-configuration runs).

### 3.1 Consequence for `STAGE2_FINDINGS.md` §2

`STAGE2_FINDINGS.md` §2 reports the Stage 2 grid as reproducing "cleanly across two independent
seeds." The seeds are independent in *classifier training only*. In
`DDSM_Colab-seed1run.ipynb`, `generate_synthetic.py` is invoked once per session (cells 20 and 32,
both `--seed 1`) and every classifier run in the grid reads those same two directories; the seed-0 and
seed-1 classifier runs all executed inside one 50-minute window (10:56–11:46 UTC, 2026-08-22),
i.e. one Colab session, one pool per arm.

So the grid varies the ~1.6-point noise term across two levels and holds the ~5-point term at **n = 1**.
The 4.11-point unfrozen SA-baseline-vs-SA-improved gap is not established to sit outside the noise
envelope, because the dominant noise dimension was never sampled. The finding's status should be
"consistent in direction across two classifier seeds at a single pool draw," not "reproducible."

---

## 4. The unfrozen reversal: probably real, with a mechanism that fits the evidence

Unlike §2's result, the unfrozen reversal does not rest on a single outlier run:

| Unfrozen (uf=2, head_bn, backbone-lr 1e-5) | seed 0 | seed 1 | mean | vs AD |
|---|---|---|---|---|
| AD (real only) | 69.84 (`un235dfj`) | 69.05 (`m53jdvq1`) | 69.44 | — |
| SA, baseline GAN | 70.90 (`qmz1jili`) | 70.63 (`5y8ju407`) | **70.77** | **+1.33** |
| SA, improved GAN | 65.87 (`kse58jjf`) | 67.46 (`tmhjvs29`) | **66.66** | **−2.78** |

Direction is consistent at both seeds. Subject to §3.1's caveat, this is worth explaining.

### 4.1 The overlooked clue: training accuracy

Every unfrozen run essentially memorizes its training set — but not equally easily:

| Unfrozen run | train_acc seed 0 | train_acc seed 1 | mean |
|---|---|---|---|
| AD (1,318 real) | 0.9833 | 0.9947 | 0.9890 |
| SA, baseline GAN (1,318 + 600) | 0.9943 | 0.9974 | **0.9958** |
| SA, improved GAN (1,318 + 600) | 0.9911 | 0.9922 | **0.9916** |

Both SA arms add exactly 600 images to the same 1,318 real ones, yet the improved-GAN mixture is
consistently *harder* to fit. Adding 600 images that are trivially separable from the real ones should
*raise* train_acc (they are easy points); adding 600 that sit among the real ones should lower it
(they compete with real examples for the same region of feature space). The baseline arm shows the
first signature, the improved arm the second.

### 4.2 Proposed mechanism: manifold overlap removes quarantine-ability

Combining §4.1 with `FINDINGS.md` §4.4's PCA result — baseline synthetic forms a tight cluster
**outside** the real-malignant region; improved synthetic **overlaps** it:

> **Baseline-GAN synthetic is off-manifold, therefore trivially separable.** A high-capacity classifier
> parks it in its own region of feature space, memorizes its labels there (train_acc 0.996), and never
> perturbs the decision boundary where real test points live. It behaves as a harmless regularizer /
> extra gradient signal → +1.33 pts.
>
> **Improved-GAN synthetic is on-manifold**, and carries partial-but-imperfect class signal
> (synthetic-only probe: 0.231 malignant recall vs the baseline pool's 0.109). Because it occupies the
> *same* feature region as real data, the classifier cannot fit it without deforming the boundary
> exactly where real test examples live. Wherever a synthetic image's conditioning label is subtly
> wrong for its position on the real manifold, that deformation costs real test accuracy →
> −2.78 pts. The depressed train_acc (0.9916) is that conflict made visible.
>
> **This can only bite when the backbone is unfrozen.** The frozen configuration trains ~33K head
> parameters on fixed ImageNet features — not enough capacity to contort around 600 on-manifold
> points. This predicts the reversal should vanish under the frozen classifier, and it does: the
> frozen SA rows (§2) sit at 65.34 vs 64.81, indistinguishable, regardless of pool.

Stated compactly: **the baseline GAN's synthetic data is safe *because* it is bad enough to be
detectable. Improving realism removed the classifier's ability to quarantine it.** Generation quality
and augmentation value are not merely "related but not interchangeable" (`FINDINGS.md` §6) — over this
range they are *anti*-correlated, for a specific and testable reason.

### 4.3 Relationship to the rejected fingerprint hypothesis

`STAGE2_FINDINGS.md` §3 tested and correctly rejected the hypothesis that the improved pool carries a
*more* learnable GAN-specific signature the classifier exploits as a shortcut. The mechanism above is
the same underlying variable — real/synthetic discriminability — with the **opposite sign**: the
improved pool is *less* distinguishable from real data, and that removes a protective separability
that was silently doing useful work for the baseline pool.

This also explains why the synthetic-only probe could not detect it. The probe evaluates each pool in
isolation, and the mechanism is defined by how synthetic points sit *relative to the real training
data* — it exists only in the mixture. That is precisely the "interaction effect between the real
training set and the specific improved-GAN synthetic distribution" §3 concluded on; this document
gives that interaction a concrete, falsifiable form.

Note also that the probe's own numbers already lean this way: the baseline-only probe collapses to the
majority class (0.109 malignant recall, train_acc 1.000 — perfect memorization, zero transfer), which
is the profile of a pool that is easy to fit and carries no real-manifold information. That is exactly
the pool that turns out to be *harmless* in the mixture.

---

## 5. Recommended next experiments

Ordered by information gained per GPU-hour.

1. **Real-vs-synthetic discriminability probe.** Train a binary real/fake classifier on each pool
   against the real training set. This directly measures the variable §4.2 proposes. Prediction:
   baseline pool → near-perfect separation; improved pool → measurably lower. Cheap, decisive, and
   currently absent from the codebase.
2. **Vary the pool seed, not just the classifier seed.** 3 pool draws × 2 classifier seeds per arm.
   Per §3, this is a prerequisite before any accuracy comparison in this project can be reported as a
   finding. `multiseed.py` currently sweeps only the classifier seed and would need extending.
3. **Run the improved ep600 pool in the unfrozen mixture** (never done — it exists only as a
   synthetic-only probe, `oy40wpcg`). It scores far better standalone than ep300 (65.08% vs 59.79%,
   malignant recall 0.340 vs 0.231). A "signal quality" account predicts it helps; the §4.2
   manifold-overlap account predicts it hurts *more*, since more training makes it more on-manifold.
   One run, cleanly discriminates the two explanations.
4. **Ratio sweep** (already proposed in `STAGE2_FINDINGS.md` §5). Under §4.2 the improved-GAN
   degradation should scale with synthetic fraction while the baseline-GAN benefit saturates.
5. **Self-labeling control.** Relabel improved-GAN synthetic with a real-data-trained classifier's own
   predictions instead of the GAN's conditioning label. If §4.2 is right — the damage is label
   conflict on the real manifold — this should recover most of the loss.

---

## 6. Required document corrections

| Document | Section | Action |
|---|---|---|
| `FINDINGS.md` | §5 (matched-epoch comparison) | Retract. Replace the +6.9 claim with the seeded −0.52 result and the §2.1 refutation of the checkpoint explanation. |
| `FINDINGS.md` | §4.5 | The +3.7-point claim shares the same uncontrolled z-draw term; downgrade to provisional. |
| `FINDINGS.md` | §6 (Discussion) | Remove "the strongest, most controlled result in the project" framing, which refers to the retracted §5. |
| `FINDINGS.md` | §3.2 | The unexplained 61.79–66.40% AD spread is now explained (§3); cross-reference. |
| `STAGE2_FINDINGS.md` | §2 | Qualify "reproduces cleanly across two independent seeds" per §3.1 — one pool draw per arm. |
| `STAGE2_FINDINGS.md` | §3 | Add §4.2/§4.3 as the concrete form of the interaction effect left open there. |
| `CovidGAN-Pytorch/REPORT.md` | §5.2, §6 | Both cite the DDSM Stage 2 numbers, including the retracted comparison. |

Reproduce this analysis:

```bash
python - <<'PY'
import wandb
for r in wandb.Api().runs("maya-hayat-ariel-university/ddsm-acgan"):
    print(r.id, r.name, r.config.get("seed"), r.summary.get("test_accuracy"))
PY
```