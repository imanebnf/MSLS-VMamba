import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [11, 22, 33, 44, 55]
MODELS = {"vmamba_baseline_strong_384": "VMamba", "current_full_strong_384": "MSLS-VMamba"}
N_BOOT = 10_000


def identify(csv_dir, summaries_dir):
    stored = {}
    for f in glob.glob(str(Path(summaries_dir) / "*final_locked_summary*.json")):
        j = json.load(open(f))
        if j["name"] in MODELS and int(j["seed"]) in SEEDS:
            t = j["plantseg_official_test"]
            stored[(j["name"], int(j["seed"]))] = dict(zip(t["paths"], t["predictions"]))
    assert len(stored) == 10
    found = {}
    for f in sorted(glob.glob(str(Path(csv_dir) / "*.csv"))):
        frame = pd.read_csv(f)
        if "pred_original" not in frame:
            continue
        hits = [k for k, pred in stored.items()
                if all(int(pred[r]) == int(p) for r, p in zip(frame.record_id, frame.pred_original))]
        assert len(hits) == 1, (f, hits)
        assert hits[0] not in found, ("two files for one run", hits[0])
        found[hits[0]] = (f, frame)
    assert len(found) == 10, sorted(found)
    return found


def main():
    csv_dir, summaries_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    (out / "raw").mkdir(parents=True, exist_ok=True)
    found = identify(csv_dir, summaries_dir)
    for (name, seed), (f, _) in found.items():
        shutil.copy(f, out / "raw" / f"{name}_seed{seed}.csv")
    ids = found[("current_full_strong_384", 11)][1].record_id.tolist()
    data = {}
    for key, (_, frame) in found.items():
        frame = frame.set_index("record_id").loc[ids]
        data[key] = frame
    base = data[("current_full_strong_384", 11)]
    subsets = {"primary": base.primary_set.astype(bool).to_numpy(),
               "fallback": (base.fallback_available.astype(bool) & (base.pred_fallback_blurred >= 0)).to_numpy()}
    second = {"primary": "region_blurred", "fallback": "fallback_blurred"}
    for key, frame in data.items():
        assert (frame.primary_set.astype(bool).to_numpy() == subsets["primary"]).all(), key

    rng = np.random.default_rng(20261114)
    rows, boot_rows = [], []
    for subset, mask in subsets.items():
        n = int(mask.sum())
        draws = [(rng.integers(0, n, n), rng.choice(len(SEEDS), size=len(SEEDS), replace=True)) for _ in range(N_BOOT)]
        per = {}
        for name in MODELS:
            m = {}
            for metric in ("lesion_specific_drop", "flip_lesion", "flip_region", "flip_difference", "acc_lesion", "acc_region"):
                m[metric] = np.zeros((len(SEEDS), n))
            for i, s in enumerate(SEEDS):
                f = data[(name, s)][mask]
                lab = f.label.to_numpy()
                m["lesion_specific_drop"][i] = (f[f"p_true_{second[subset]}"] - f.p_true_lesion_blurred).to_numpy()
                m["flip_lesion"][i] = (f.pred_lesion_blurred != f.pred_original).to_numpy()
                m["flip_region"][i] = (f[f"pred_{second[subset]}"] != f.pred_original).to_numpy()
                m["flip_difference"][i] = m["flip_lesion"][i] - m["flip_region"][i]
                m["acc_lesion"][i] = (f.pred_lesion_blurred.to_numpy() == lab)
                m["acc_region"][i] = (f[f"pred_{second[subset]}"].to_numpy() == lab)
            per[name] = m
        for metric in per["current_full_strong_384"]:
            stats = {}
            for name, label in list(MODELS.items()) + [("difference", "MSLS-VMamba minus VMamba")]:
                if name == "difference":
                    arr = per["current_full_strong_384"][metric] - per["vmamba_baseline_strong_384"][metric]
                else:
                    arr = per[name][metric]
                point = float(arr.mean())
                seed_means = arr.mean(axis=1)
                b = np.array([arr[sd][:, idx].mean() for idx, sd in draws])
                lo, hi = np.percentile(b, [2.5, 97.5])
                rows.append({"subset": subset, "n_images": n, "metric": metric, "model": label, "estimate": point,
                             **{f"seed{s}": float(v) for s, v in zip(SEEDS, seed_means)},
                             "boot_ci95_low": float(lo), "boot_ci95_high": float(hi),
                             "positive_seeds": int((seed_means > 0).sum())})
    res = pd.DataFrame(rows)
    res.to_csv(out / "lesion_perturbation_results.csv", index=False)
    show = res[res.metric.isin(["lesion_specific_drop", "flip_lesion", "flip_region", "flip_difference"])]
    for r in show.itertuples(index=False):
        print(f"{r.subset:<9} n={r.n_images:<4} {r.metric:<22} {r.model:<26} {r.estimate:+.4f} "
              f"[{r.boot_ci95_low:+.4f}, {r.boot_ci95_high:+.4f}]  +seeds {r.positive_seeds}/5")


if __name__ == "__main__":
    main()
