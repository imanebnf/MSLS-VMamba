import glob
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
SPLITS = {"plantseg_official_test": "PlantSeg test", "plantdoc_external_clean": "PlantDoc"}
LABELS = {"current_full_strong_384": "MSLS-VMamba", "vmamba_baseline_strong_384": "VMamba",
          "efficientnet_v2_s_384": "EfficientNetV2-S", "resnet50_384": "ResNet50", "vit_s16_384": "ViT-S/16"}
SEEDS = (11, 22, 33, 44, 55)
MSV = "MSVMamba-Tiny"
N_BOOT = 10_000
CLASS_NAMES = json.load(open(ROOT / "data/cleaning_protocol.json"))["class_names"]


def ci(values):
    values = np.asarray(values, dtype=float)
    n, mean, sd = len(values), float(values.mean()), float(values.std(ddof=1))
    half = float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n) if n > 1 else float("nan")
    return mean, sd, mean - half, mean + half, int((values > 0).sum())


summaries = {}
for path in (glob.glob(str(ROOT / "runs/vmamba_msls_seeds11-55/*final_locked_summary.json"))
             + glob.glob(str(ROOT / "runs/baselines_seeds11-55/*final_locked_summary.json"))):
    s = json.load(open(path))
    if s["name"] in LABELS and int(s["seed"]) in SEEDS:
        key = (s["name"], int(s["seed"]))
        assert key not in summaries, f"duplicate summary {key}"
        summaries[key] = s
assert len(summaries) == 25, len(summaries)

metric_rows, class_rows = [], []
for (name, seed), s in summaries.items():
    for split, label in SPLITS.items():
        y, p = np.asarray(s[split]["targets"]), np.asarray(s[split]["predictions"])
        f1 = 100 * f1_score(y, p, average="macro", zero_division=0)
        assert abs(f1 - float(s[split]["macro_f1"])) < 1e-6, (name, seed, split)
        metric_rows.append({"model": LABELS[name], "seed": seed, "split": label,
                            "accuracy": 100 * accuracy_score(y, p), "balanced_accuracy": 100 * balanced_accuracy_score(y, p),
                            "macro_f1": f1, "n": len(y)})
        per_class = 100 * f1_score(y, p, average=None, labels=list(range(16)), zero_division=0)
        for k, value in enumerate(per_class):
            class_rows.append({"model": LABELS[name], "seed": seed, "split": label, "class": CLASS_NAMES[k],
                               "class_index": k + 1, "f1": value, "support": int((y == k).sum())})
msv = pd.read_csv(ROOT / "analysis_outputs/msvmamba_comparison/msvmamba_per_seed.csv")
msv = msv[(msv.label == MSV) & msv.split.isin(SPLITS)]
sizes = {row["split"]: row["n"] for row in metric_rows}
for row in msv.itertuples(index=False):
    metric_rows.append({"model": MSV, "seed": int(row.seed), "split": SPLITS[row.split], "accuracy": row.accuracy,
                        "balanced_accuracy": row.balanced_accuracy, "macro_f1": row.macro_f1,
                        "n": sizes[SPLITS[row.split]]})
metrics = pd.DataFrame(metric_rows).sort_values(["split", "model", "seed"])
assert len(metrics) == 60, len(metrics)
metrics.to_csv(OUT / "S2_per_seed_metrics.csv", index=False)
per_class = pd.DataFrame(class_rows).sort_values(["split", "model", "seed"], kind="stable")
per_class.to_csv(OUT / "S4_per_class_f1_by_seed.csv", index=False)

paired_rows = []
for split in ("PlantSeg test", "PlantDoc"):
    pivot = metrics[metrics.split == split].pivot(index="seed", columns="model", values="macro_f1")
    for comparator in ("VMamba", "EfficientNetV2-S", "ResNet50", "ViT-S/16", MSV):
        d = (pivot["MSLS-VMamba"] - pivot[comparator]).to_numpy()
        mean, sd, low, high, positive = ci(d)
        paired_rows.append({"split": split, "comparison": f"MSLS-VMamba - {comparator}",
                            **{f"seed_{s}": v for s, v in zip(SEEDS, d)},
                            "mean_difference": mean, "sd": sd, "t_ci95_low": low, "t_ci95_high": high,
                            "positive_seeds": positive,
                            "difference_of_rounded_means": round(pivot["MSLS-VMamba"].mean(), 2) - round(pivot[comparator].mean(), 2)})
paired = pd.DataFrame(paired_rows)
assert len(paired) == 10, len(paired)
paired.to_csv(OUT / "S3_paired_differences_unrounded.csv", index=False)

ablation = pd.read_csv(ROOT / "analysis_outputs/seed11/seed11_ablation.csv")
ablation.to_csv(OUT / "S3_table6_ablation_unrounded.csv", index=False)

cls_rows, loo_rows = [], []
for split in ("PlantDoc", "PlantSeg test"):
    sub = per_class[per_class.split == split]
    for comparator in ("VMamba", "EfficientNetV2-S"):
        for name in CLASS_NAMES:
            a = sub[(sub.model == "MSLS-VMamba") & (sub["class"] == name)].set_index("seed").f1.reindex(SEEDS)
            b = sub[(sub.model == comparator) & (sub["class"] == name)].set_index("seed").f1.reindex(SEEDS)
            mean, sd, low, high, positive = ci((a - b).to_numpy())
            cls_rows.append({"split": split, "comparison": f"MSLS-VMamba - {comparator}", "class": name,
                             "support": int(sub[(sub["class"] == name)].support.iloc[0]),
                             "msls_mean_f1": float(a.mean()), "comparator_mean_f1": float(b.mean()),
                             "mean_difference": mean, "t_ci95_low": low, "t_ci95_high": high, "positive_seeds": positive})
        for drop in [None] + CLASS_NAMES:
            keep = [c for c in CLASS_NAMES if c != drop]
            a = sub[(sub.model == "MSLS-VMamba") & sub["class"].isin(keep)].groupby("seed").f1.mean().reindex(SEEDS)
            b = sub[(sub.model == comparator) & sub["class"].isin(keep)].groupby("seed").f1.mean().reindex(SEEDS)
            mean, sd, low, high, positive = ci((a - b).to_numpy())
            loo_rows.append({"split": split, "comparison": f"MSLS-VMamba - {comparator}", "class_left_out": drop or "(none)",
                             "mean_difference": mean, "t_ci95_low": low, "t_ci95_high": high, "positive_seeds": positive})
per_class_paired = pd.DataFrame(cls_rows)
per_class_paired.to_csv(OUT / "S4_per_class_paired_differences.csv", index=False)
loo = pd.DataFrame(loo_rows)
loo.to_csv(OUT / "S4_leave_one_class_out.csv", index=False)

MSLS, VM = "current_full_strong_384", "vmamba_baseline_strong_384"
RAW = ROOT / "runs/alignment_interventions_test_seeds11-55"
frames = {(name, seed): pd.read_csv(RAW / f"{name}_seed{seed}_test_alignment_interventions.csv")
          for name in (MSLS, VM) for seed in SEEDS}
ids = frames[(MSLS, SEEDS[0])].record_id.tolist()
assert all(frame.record_id.tolist() == ids for frame in frames.values())
y = frames[(MSLS, SEEDS[0])].label.to_numpy()
n_img, n_seed = len(y), len(SEEDS)
rng = np.random.default_rng(20261114)
draws = [(rng.integers(0, n_img, n_img), rng.choice(n_seed, size=n_seed, replace=True)) for _ in range(N_BOOT)]
seed_resamples = np.array(list(itertools.product(range(n_seed), repeat=n_seed)))


def interval(values):
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def level_row(outcome, unit, per_seed, joint, image_only):
    per_seed = np.asarray(per_seed, dtype=float)
    seed_only = per_seed[seed_resamples].mean(1)
    (j_lo, j_hi), (i_lo, i_hi), (s_lo, s_hi) = interval(joint), interval(image_only), interval(seed_only)
    return {"outcome": outcome, "unit": unit, "mean_difference": float(per_seed.mean()),
            "images_and_seeds_ci95_low": j_lo, "images_and_seeds_ci95_high": j_hi,
            "image_only_ci95_low": i_lo, "image_only_ci95_high": i_hi,
            "seed_only_ci95_low": s_lo, "seed_only_ci95_high": s_hi,
            **{f"seed_{s}": v for s, v in zip(SEEDS, per_seed)}}


level_rows = []
for column, outcome, unit, scale in (
        ("layercam_energy", "LayerCAM share inside lesion, MSLS-VMamba - VMamba", "pp", 100.0),
        ("layercam_pointing", "LayerCAM peak inside lesion, MSLS-VMamba - VMamba", "pp", 100.0),
        ("layercam_concentration", "LayerCAM concentration, MSLS-VMamba - VMamba", "ratio", 1.0)):
    d = scale * np.stack([frames[(MSLS, s)][column].to_numpy() - frames[(VM, s)][column].to_numpy() for s in SEEDS])
    level_rows.append(level_row(outcome, unit, d.mean(1),
                                [d[sd][:, idx].mean() for idx, sd in draws],
                                [d[:, idx].mean() for idx, _ in draws]))


def macro_f1(targets, predictions):
    return 100 * f1_score(targets, predictions, average="macro", zero_division=0)


full = np.stack([frames[(MSLS, s)]["pred_full"].to_numpy() for s in SEEDS])
removed = np.stack([frames[(MSLS, s)]["pred_global_only"].to_numpy() for s in SEEDS])
drop = np.array([macro_f1(y, full[k]) - macro_f1(y, removed[k]) for k in range(n_seed)])
joint, image_only = np.empty(N_BOOT), np.empty(N_BOOT)
for b, (idx, sd) in enumerate(draws):
    d = np.array([macro_f1(y[idx], full[k][idx]) - macro_f1(y[idx], removed[k][idx]) for k in range(n_seed)])
    joint[b], image_only[b] = d[sd].mean(), d.mean()
level_rows.append(level_row("Macro-F1 drop after descriptor removal, MSLS-VMamba", "pp", drop, joint, image_only))
pd.DataFrame(level_rows).to_csv(OUT / "S6_paired_differences_bootstrap_by_source.csv", index=False)

print("wrote", OUT)
