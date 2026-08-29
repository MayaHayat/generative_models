"""
Stage 2 -- assemble the paper / reconstruction / improved comparison table
required by the report ("Include a table with the original results, your
reconstruction results and the results of your improved model").

Reads the curated Stage 2 multi-seed aggregates in stage2/results/ (written by
stage2/multiseed.py, one per unfreeze-block variant) and prints the comparison,
showing BOTH improved variants (uf=1 and uf=2). The paper and Stage 1
reconstruction figures are constants below, sourced from the paper and
FINDINGS.md (§3, §8) respectively -- fixed, already-reported numbers.

    python -m stage2.compare_results
"""
import argparse
import json
from pathlib import Path

# Paper (Waheed et al. 2020, Table 1) and our Stage 1 reconstruction
# (FINDINGS.md §3 / §8.2). Accuracy as fractions.
PAPER = {"ad": 0.85, "sa": 0.95}
STAGE1 = {"ad": 0.9062, "sa": 0.9010}  # frozen VGG16, single run


def _load(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d["ad"]["accuracy"], d["sa"]["accuracy"], d["ad"]["n_runs"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="stage2/results")
    args = ap.parse_args()
    rd = Path(args.results_dir)

    rows = [
        ("Paper (Waheed et al. 2020)", PAPER["ad"], PAPER["sa"], None, "single run"),
        ("Stage 1 reconstruction (frozen VGG16)", STAGE1["ad"], STAGE1["sa"], None, "single run"),
    ]
    spreads = []
    for label, fname in [("Stage 2 improved / uf=1 (1 block + BN)", "multiseed_uf1.json"),
                         ("Stage 2 improved / uf=2 (2 blocks + BN)", "multiseed_uf2.json")]:
        p = rd / fname
        if not p.exists():
            print(f"(skipping {label}: missing {p})")
            continue
        ad, sa, n = _load(p)
        rows.append((label, ad["mean"], sa["mean"], sa["std"], f"mean of {n} seeds"))
        spreads.append((label, ad, sa, n))

    w = 40
    print("=" * 92)
    print("COVID-19 detector: paper vs reconstruction vs improved model")
    print("=" * 92)
    print(f"{'Model':<{w}}{'CNN-AD':>12}{'CNN-SA':>12}{'aug. lift':>12}   notes")
    print("-" * 92)
    for name, ad, sa, _sastd, note in rows:
        lift = f"{(sa-ad)*100:+.2f}"
        print(f"{name:<{w}}{ad*100:>11.2f}%{sa*100:>11.2f}%{lift:>12}   {note}")
    print("-" * 92)
    for label, ad, sa, n in spreads:
        print(f"{label}:  CNN-AD {ad['mean']*100:.2f}% +/- {ad['std']*100:.2f}   "
              f"CNN-SA {sa['mean']*100:.2f}% +/- {sa['std']*100:.2f}   (pstdev over {n} seeds)")
    print("\nReading: the paper shows a +10pt lift from an 85% baseline; our frozen Stage 1")
    print("reconstruction had no headroom (~90.6%) so augmentation was flat; Stage 2 unfreezes")
    print("the top VGG16 block(s), lifting the baseline AND restoring a positive augmentation")
    print("effect. More unlocked capacity (uf=2) helps both the baseline and the lift, with")
    print("CNN-SA (96.67%) edging past the paper's 95%.")


if __name__ == "__main__":
    main()
