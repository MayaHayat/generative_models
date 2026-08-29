"""
Stage 2 -- generate the report figures from the curated results.

Reads stage2/results/ (and the GAN sample PNGs under runs/) and writes five
figures to stage2/results/figures/:

  1. generalization_curve.png  -- per-epoch train/test accuracy, AD & SA (Test 8)
  2. data_scarcity.png         -- CNN-AD/CNN-SA vs real-data fraction (Test 7)
  3. comparison_bar.png        -- paper / reconstruction / improved (headline)
  4. fid_drop.png              -- FID 504 -> 273 by class (Test 5)
  5. gan_samples.png           -- generator samples at epochs 10 -> 2000

    python -m stage2.make_figures
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("stage2/results")
FIGS = RESULTS / "figures"
AD_C = "#2563eb"   # blue  = real-only (AD)
SA_C = "#dc2626"   # red   = +synthetic (SA)


def _load_curve(mode):
    rows = list(csv.DictReader(open(RESULTS / f"diagnostic_curve_{mode}.csv")))
    ep = [int(r["epoch"]) for r in rows]
    tr = [float(r["train_acc"]) * 100 for r in rows]
    te = [float(r["test_acc"]) * 100 for r in rows]
    return ep, tr, te


def fig_generalization():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, mode, title in [(axes[0], "ad", "CNN-AD (real only)"),
                            (axes[1], "sa", "CNN-SA (real + synthetic)")]:
        ep, tr, te = _load_curve(mode)
        ax.plot(ep, tr, "--", color="#6b7280", label="train")
        ax.plot(ep, te, "-o", color=(AD_C if mode == "ad" else SA_C), ms=3, label="test")
        ax.axvline(15, color="#9ca3af", ls=":", lw=1)
        ax.text(15.3, 60, "reported\n(15 ep)", fontsize=7, color="#6b7280")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylim(55, 101)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("accuracy (%)")
    fig.suptitle("Generalization curve — train saturates, test plateaus (no overfitting)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "generalization_curve.png", dpi=150)
    plt.close(fig)


def fig_scarcity():
    d = json.load(open(RESULTS / "scarcity_uf2.json"))
    pts = sorted(d["points"], key=lambda p: p["fraction"])
    fr = [p["fraction"] * 100 for p in pts]
    ad = [p["ad_accuracy"]["mean"] * 100 for p in pts]
    sa = [p["sa_accuracy"]["mean"] * 100 for p in pts]
    lift = [p["accuracy_lift"] * 100 for p in pts]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(fr, ad, "-o", color=AD_C, label="CNN-AD (real only)")
    ax.plot(fr, sa, "-o", color=SA_C, label="CNN-SA (+ synthetic)")
    for x, a, s, l in zip(fr, ad, sa, lift):
        ax.annotate(f"+{l:.1f}", (x, (a + s) / 2), fontsize=8, color="#059669",
                    ha="center", va="center")
    ax.set_xlabel("real training data used (%)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Data-scarcity curve — augmentation helps at every level (uf=2)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "data_scarcity.png", dpi=150)
    plt.close(fig)


def fig_comparison():
    uf1 = json.load(open(RESULTS / "multiseed_uf1.json"))
    uf2 = json.load(open(RESULTS / "multiseed_uf2.json"))
    models = ["Paper", "Stage 1\n(frozen)", "Stage 2\nuf=1", "Stage 2\nuf=2"]
    ad = [85.0, 90.62, uf1["ad"]["accuracy"]["mean"] * 100, uf2["ad"]["accuracy"]["mean"] * 100]
    sa = [95.0, 90.10, uf1["sa"]["accuracy"]["mean"] * 100, uf2["sa"]["accuracy"]["mean"] * 100]

    x = range(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar([i - w / 2 for i in x], ad, w, color=AD_C, label="CNN-AD (real only)")
    b2 = ax.bar([i + w / 2 for i in x], sa, w, color=SA_C, label="CNN-SA (+ synthetic)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                    f"{b.get_height():.1f}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(80, 100)
    ax.set_title("Paper vs reconstruction vs improved — the augmentation effect")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "comparison_bar.png", dpi=150)
    plt.close(fig)


def fig_fid():
    labels = ["overall", "COVID", "Normal"]
    smoke = [504.4, 487.2, 538.0]
    full = [272.7, 302.2, 290.2]
    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar([i - w / 2 for i in x], smoke, w, color="#9ca3af", label="25-epoch (near-noise)")
    ax.bar([i + w / 2 for i in x], full, w, color="#059669", label="2000-epoch (trained)")
    for i, (a, b) in enumerate(zip(smoke, full)):
        ax.text(i - w / 2, a + 6, f"{a:.0f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 6, f"{b:.0f}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("FID  (lower = more realistic)")
    ax.set_title("Generator quality: FID roughly halved over training")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fid_drop.png", dpi=150)
    plt.close(fig)


def fig_gan_samples():
    from PIL import Image
    epochs = [10, 100, 500, 1000, 2000]
    paths = [Path(f"runs/gan/samples/epoch_{e:04d}.png") for e in epochs]
    have = [(e, p) for e, p in zip(epochs, paths) if p.exists()]
    if not have:
        print("  (skip gan_samples: no runs/gan/samples/*.png found)")
        return
    fig, axes = plt.subplots(1, len(have), figsize=(2.4 * len(have), 2.8))
    if len(have) == 1:
        axes = [axes]
    for ax, (e, p) in zip(axes, have):
        ax.imshow(Image.open(p))
        ax.set_title(f"epoch {e}", fontsize=9)
        ax.axis("off")
    fig.suptitle("AC-GAN samples over training (COVID/Normal grid)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "gan_samples.png", dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_generalization()
    fig_scarcity()
    fig_comparison()
    fig_fid()
    fig_gan_samples()
    print("figures written to", FIGS)
    for p in sorted(FIGS.glob("*.png")):
        print("  ", p)


if __name__ == "__main__":
    main()
