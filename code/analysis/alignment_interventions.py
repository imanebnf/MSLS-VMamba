import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SEEDS = [11, 22, 33, 44, 55]
MSLS, VM = "current_full_strong_384", "vmamba_baseline_strong_384"
N_BOOT = 10_000


def main():
    raw, summaries, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    locked = {}
    for f in glob.glob(str(summaries / "*final_locked_summary*.json")):
        j = json.load(open(f))
        if j["name"] in (MSLS, VM):
            t = j["plantseg_official_test"]
            locked[(j["name"], int(j["seed"]))] = dict(zip(t["paths"], t["predictions"]))
    data = {}
    for name in (MSLS, VM):
        for s in SEEDS:
            frame = pd.read_csv(raw / f"{name}_seed{s}_test_alignment_interventions.csv")
            assert all(int(locked[(name, s)][r]) == int(p) for r, p in zip(frame.record_id, frame.pred)), (name, s)
            data[(name, s)] = frame
    ids = data[(MSLS, 11)].record_id.tolist()
    n = len(ids)
    for key, frame in data.items():
        assert frame.record_id.tolist() == ids, key
    y = data[(MSLS, 11)].label.to_numpy()
    rng = np.random.default_rng(20261114)
    draws = [(rng.integers(0, n, n), rng.choice(len(SEEDS), size=len(SEEDS), replace=True)) for _ in range(N_BOOT)]

    def boot_mean(arr):
        return np.array([arr[sd][:, idx].mean() for idx, sd in draws])

    def boot_f1(pred, mask=None):
        vals = np.empty(N_BOOT)
        yy, pp = (y, pred) if mask is None else (y[mask], pred[:, mask])
        local = draws if mask is None else [(np.random.default_rng(20261116 + b).integers(0, len(yy), len(yy)), sd)
                                             for b, (_, sd) in enumerate(draws)]
        for b, (idx, sd) in enumerate(local):
            vals[b] = np.mean([100 * f1_score(yy[idx], pp[s][idx], average="macro", zero_division=0) for s in sd])
        return vals

    rows = []

    def add(name, quantity, arr_point, boot, per_seed):
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"model": name, "quantity": quantity, "estimate": float(arr_point), "boot_ci95_low": float(lo),
                     "boot_ci95_high": float(hi), **{f"seed{s}": float(v) for s, v in zip(SEEDS, per_seed)}})

    for metric in ("layercam_energy", "layercam_concentration", "layercam_pointing"):
        arrs = {name: np.stack([data[(name, s)][metric].to_numpy() for s in SEEDS]) for name in (MSLS, VM)}
        for name, label in ((VM, "VMamba"), (MSLS, "MSLS-VMamba")):
            add(label, metric, arrs[name].mean(), boot_mean(arrs[name]), arrs[name].mean(1))
        diff = arrs[MSLS] - arrs[VM]
        add("MSLS-VMamba minus VMamba", metric, diff.mean(), boot_mean(diff), diff.mean(1))
    has_within = data[(MSLS, 11)].has_within_crop_donor.astype(bool).to_numpy()
    for cond in ("full", "global_only", "regional_only", "cross_class_donor", "within_crop_donor"):
        pred = np.stack([data[(MSLS, s)][f"pred_{cond}"].to_numpy() for s in SEEDS])
        mask = has_within if cond == "within_crop_donor" else None
        per_seed = [100 * f1_score(y[mask] if mask is not None else y, p[mask] if mask is not None else p,
                                   average="macro", zero_division=0) for p in pred]
        add("MSLS-VMamba", f"macro_f1_{cond}", np.mean(per_seed), boot_f1(pred, mask), per_seed)
    m = int(has_within.sum())
    rng_w = np.random.default_rng(20261115)
    draws_w = [(rng_w.integers(0, m, m), sd) for _, sd in draws]

    def boot_mean_w(arr):
        return np.array([arr[sd][:, idx].mean() for idx, sd in draws_w])

    for cond, donor_col in (("cross_class_donor", "cross_donor_label"), ("within_crop_donor", "within_donor_label")):
        follow = np.stack([(data[(MSLS, s)][f"pred_{cond}"] == data[(MSLS, s)][donor_col]).to_numpy(float) for s in SEEDS])
        keep = np.stack([(data[(MSLS, s)][f"pred_{cond}"] == data[(MSLS, s)].label).to_numpy(float) for s in SEEDS])
        if cond == "within_crop_donor":
            follow, keep = follow[:, has_within], keep[:, has_within]
            add("MSLS-VMamba", f"follow_donor_{cond}", follow.mean(), boot_mean_w(follow), follow.mean(1))
            add("MSLS-VMamba", f"keep_true_{cond}", keep.mean(), boot_mean_w(keep), keep.mean(1))
        else:
            add("MSLS-VMamba", f"follow_donor_{cond}", follow.mean(), boot_mean(follow), follow.mean(1))
            add("MSLS-VMamba", f"keep_true_{cond}", keep.mean(), boot_mean(keep), keep.mean(1))
    res = pd.DataFrame(rows)
    res.to_csv(out / "alignment_interventions_results.csv", index=False)
    for r in res.itertuples(index=False):
        print(f"{r.model:<26} {r.quantity:<34} {r.estimate:8.3f} [{r.boot_ci95_low:8.3f}, {r.boot_ci95_high:8.3f}]")


if __name__ == "__main__":
    main()
