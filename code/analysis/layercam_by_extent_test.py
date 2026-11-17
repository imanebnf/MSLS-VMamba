import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [11, 22, 33, 44, 55]
MODELS = {"vmamba_baseline_strong_384": "VMamba", "current_full_strong_384": "MSLS-VMamba"}
BINS = [-1e-9, 0.05, 0.15, 0.30, 1.01]
LABELS = ["<5%", "5–15%", "15–30%", "≥30%"]
N_BOOT = 10_000


def main():
    raw, regions_csv, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    data = {(name, s): pd.read_csv(raw / f"{name}_seed{s}_test_alignment_interventions.csv") for name in MODELS for s in SEEDS}
    ids = data[("current_full_strong_384", 11)].record_id.tolist()
    assert len(ids) == 435 and all(f.record_id.tolist() == ids for f in data.values())
    regions = pd.read_csv(regions_csv).set_index("record_id").loc[ids]
    area = regions.lesion_pixels_384.to_numpy() / float(384 * 384)
    worst = 0.0
    for key, f in data.items():
        ok = (f.layercam_concentration.to_numpy() > 0) & (f.layercam_energy.to_numpy() > 0)
        implied = f.layercam_energy.to_numpy()[ok] / f.layercam_concentration.to_numpy()[ok]
        worst = max(worst, float(np.max(np.abs(implied - area[ok]) / area[ok])))
    assert worst < 1e-3, worst
    print(f"lesion area check: regions file vs energy/concentration, largest relative gap {worst:.2e}")
    group = pd.cut(area, bins=BINS, labels=LABELS)
    conc = {name: np.stack([data[(name, s)].layercam_concentration.to_numpy() for s in SEEDS]) for name in MODELS}
    rng = np.random.default_rng(20261115)
    rows = []
    for lab in LABELS:
        mask = np.asarray(group == lab)
        n = int(mask.sum())
        draws = [(rng.integers(0, n, n), rng.choice(len(SEEDS), size=len(SEEDS), replace=True)) for _ in range(N_BOOT)]
        arrs = {name: conc[name][:, mask] for name in MODELS}
        arrs["difference"] = arrs["current_full_strong_384"] - arrs["vmamba_baseline_strong_384"]
        for name, arr in arrs.items():
            boot = np.array([arr[sd][:, idx].mean() for idx, sd in draws])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            seed_means = arr.mean(1)
            rows.append({"extent": lab, "model": MODELS.get(name, "MSLS-VMamba minus VMamba"), "n": n, "mean": float(arr.mean()),
                         "lo": float(lo), "hi": float(hi), **{f"seed{s}": float(v) for s, v in zip(SEEDS, seed_means)},
                         "positive_seeds": int((seed_means > 0).sum())})
    res = pd.DataFrame(rows)
    res.to_csv(out / "layercam_concentration_by_extent_test_five_seeds.csv", index=False)
    print(f"all images: VMamba {conc['vmamba_baseline_strong_384'].mean():.4f}, MSLS-VMamba {conc['current_full_strong_384'].mean():.4f}")
    for r in res.itertuples(index=False):
        print(f"{r.extent:<7} n={r.n:<4} {r.model:<26} {r.mean:6.3f} [{r.lo:6.3f}, {r.hi:6.3f}]  seeds: "
              + " ".join(f"{getattr(r, 'seed%d' % s):.2f}" for s in SEEDS) + f"  (+{r.positive_seeds}/5)")


if __name__ == "__main__":
    main()
