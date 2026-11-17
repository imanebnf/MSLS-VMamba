import glob
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
metrics = pd.DataFrame(metric_rows).sort_values(["split", "model", "seed"])
metrics.to_csv(OUT / "S2_per_seed_metrics.csv", index=False)
per_class = pd.DataFrame(class_rows).sort_values(["split", "model", "seed"], kind="stable")
per_class.to_csv(OUT / "S4_per_class_f1_by_seed.csv", index=False)

paired_rows = []
for split in ("PlantSeg test", "PlantDoc"):
    pivot = metrics[metrics.split == split].pivot(index="seed", columns="model", values="macro_f1")
    for comparator in ("VMamba", "EfficientNetV2-S", "ResNet50", "ViT-S/16"):
        d = (pivot["MSLS-VMamba"] - pivot[comparator]).to_numpy()
        mean, sd, low, high, positive = ci(d)
        paired_rows.append({"split": split, "comparison": f"MSLS-VMamba - {comparator}",
                            **{f"seed_{s}": v for s, v in zip(SEEDS, d)},
                            "mean_difference": mean, "sd": sd, "t_ci95_low": low, "t_ci95_high": high,
                            "positive_seeds": positive,
                            "difference_of_rounded_means": round(pivot["MSLS-VMamba"].mean(), 2) - round(pivot[comparator].mean(), 2)})
paired = pd.DataFrame(paired_rows)
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

print("wrote", OUT)
