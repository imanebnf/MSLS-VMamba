import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

SEEDS = [11, 22, 33, 44, 55]
LABEL = {"vmamba_baseline": "VMamba", "current_full": "MSLS-VMamba", "vit_s16": "ViT-S/16",
         "resnet50": "ResNet50", "efficientnet_v2_s": "EfficientNetV2-S"}
ORDER = ["resnet50", "efficientnet_v2_s", "vit_s16", "vmamba_baseline", "current_full"]
SPLITS = {"plantseg_official_test": "PlantSeg test", "plantdoc_external_clean": "PlantDoc"}
# The bootstrap draws the validation split first so that the published intervals are reproduced exactly.
DRAW_ORDER = ("optimization_validation", *SPLITS)
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
N_BOOT = 10_000


def recompute(split):
    y, p = np.asarray(split["targets"]), np.asarray(split["predictions"])
    return {"accuracy": 100 * accuracy_score(y, p), "balanced_accuracy": 100 * balanced_accuracy_score(y, p),
            "macro_f1": 100 * f1_score(y, p, average="macro", zero_division=0)}


def macro_f1(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return float(np.mean(2 * tp[den > 0] / den[den > 0]))


def main():
    out = Path(sys.argv[1])
    runs = {}
    for d in sys.argv[2:]:
        for f in Path(d).glob("*final_locked_summary*.json"):
            j = json.loads(f.read_text(encoding="utf-8"))
            if j.get("model") in LABEL and int(j["seed"]) in SEEDS:
                key = (j["model"], int(j["seed"]))
                assert key not in runs, f"duplicate run {key}"
                runs[key] = j
    models = [m for m in ORDER if all((m, s) in runs for s in SEEDS)]
    incomplete = sorted({m for m, _ in runs} - set(models))
    rows = []
    for m in models:
        for s in SEEDS:
            j = runs[(m, s)]
            for sp in SPLITS:
                r = recompute(j[sp])
                for k in METRICS:
                    assert abs(r[k] - j[sp][k]) < 1e-6, (m, s, sp, k)
                rows.append({"model": m, "model_label": LABEL[m], "seed": s, "split": sp, "best_epoch": j["best_epoch"],
                             "parameters": j["parameters"], "gpu": j["runtime"]["gpu"],
                             "training_minutes": j.get("training_minutes", ""), "evaluated_utc": j.get("evaluated_utc", ""), **r})
    write(out / "main_per_seed.csv", rows)
    val = {(r["model"], r["seed"], r["split"]): r for r in rows}

    summ = [{"model": m, "split": sp, "metric": k, "mean": st.mean(xs := [val[(m, s, sp)][k] for s in SEEDS]),
             "sample_sd": st.stdev(xs), "n_seeds": 5} for m in models for sp in SPLITS for k in METRICS]
    write(out / "main_mean_sd.csv", summ)

    rng = np.random.default_rng(20261031)
    draws_by_split, msls_f1 = {}, {}
    n_cls = len(runs[("current_full", SEEDS[0])]["identity"]["class_names"])
    for sp in DRAW_ORDER:
        n = len(runs[("current_full", SEEDS[0])][sp]["targets"])
        draws_by_split[sp] = [(rng.integers(0, n, n), rng.choice(SEEDS, size=5, replace=True)) for _ in range(N_BOOT)]
    for m in models:
        for s in SEEDS:
            for sp in SPLITS:
                y, p = np.asarray(runs[(m, s)][sp]["targets"]), np.asarray(runs[(m, s)][sp]["predictions"])
                assert abs(100 * macro_f1(y, p, n_cls) - val[(m, s, sp)]["macro_f1"]) < 1e-9, ("fast macro-F1", m, s, sp)

    pair, boot = [], []
    for m in [x for x in models if x != "current_full"]:
        for sp in SPLITS:
            for k in METRICS:
                d = [val[("current_full", s, sp)][k] - val[(m, s, sp)][k] for s in SEEDS]
                mean, sd = st.mean(d), st.stdev(d)
                se = sd / 5 ** 0.5
                t = sps.t.ppf(0.975, 4)
                pair.append({"comparison": f"MSLS-VMamba minus {LABEL[m]}", "comparator": m, "split": sp, "metric": k,
                             **{f"diff_seed{s}": x for s, x in zip(SEEDS, d)}, "mean_diff": mean, "sd_diff": sd,
                             "t_ci95_low": mean - t * se, "t_ci95_high": mean + t * se,
                             "positive_seeds": sum(x > 0 for x in d)})
            y = np.asarray(runs[(m, SEEDS[0])][sp]["targets"])
            pm = {s: np.asarray(runs[("current_full", s)][sp]["predictions"]) for s in SEEDS}
            pc = {s: np.asarray(runs[(m, s)][sp]["predictions"]) for s in SEEDS}
            for s in SEEDS:
                assert runs[(m, s)][sp]["paths"] == runs[("current_full", s)][sp]["paths"], "image order differs"
            for s in SEEDS:
                assert np.array_equal(np.asarray(runs[(m, s)][sp]["targets"]), y), "targets differ"
            if sp not in msls_f1:
                msls_f1[sp] = [{s: macro_f1(y[idx], pm[s][idx], n_cls) for s in set(seeds.tolist())}
                               for idx, seeds in draws_by_split[sp]]
            draws = np.empty(N_BOOT)
            for b, (idx, seeds) in enumerate(draws_by_split[sp]):
                fc = {s: macro_f1(y[idx], pc[s][idx], n_cls) for s in set(seeds.tolist())}
                draws[b] = 100 * float(np.mean([msls_f1[sp][b][s] - fc[s] for s in seeds]))
            lo, hi = np.percentile(draws, [2.5, 97.5])
            boot.append({"comparison": f"MSLS-VMamba minus {LABEL[m]}", "comparator": m, "split": sp, "metric": "macro_f1",
                         "boot_ci95_low": lo, "boot_ci95_high": hi, "n_boot": N_BOOT})
    write(out / "main_paired_vs_msls.csv", pair)
    write(out / "main_paired_vs_msls_bootstrap.csv", boot)

    print("wrote", out)
    if incomplete:
        print("incomplete:", incomplete)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
