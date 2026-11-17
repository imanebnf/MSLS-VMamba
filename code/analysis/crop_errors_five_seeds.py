import csv
import glob
import json
import sys
from pathlib import Path

SEEDS = [11, 22, 33, 44, 55]
MODELS = {"vmamba_baseline": "VMamba", "current_full": "MSLS-VMamba"}
SPLITS = {"plantseg_official_test": "PlantSeg test", "plantdoc_external_clean": "PlantDoc"}


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for f in glob.glob(str(src / "*final_locked_summary*.json")):
        j = json.loads(Path(f).read_text(encoding="utf-8"))
        if j["model"] in MODELS and int(j["seed"]) in SEEDS:
            runs[(j["model"], int(j["seed"]))] = j
    assert len(runs) == 10, sorted(runs)
    first = runs[("current_full", 11)]
    names = (first.get("identity") or first["run_identity"]["payload"])["class_names"]
    crop = [n.split("_", 1)[0] for n in names]
    rows = []
    for split, split_label in SPLITS.items():
        for model, label in MODELS.items():
            per_seed = []
            for seed in SEEDS:
                t = runs[(model, seed)][split]
                wrong = [(a, b) for a, b in zip(t["targets"], t["predictions"]) if a != b]
                cross = sum(crop[a] != crop[b] for a, b in wrong)
                per_seed.append((len(wrong), cross, len(wrong) - cross))
                rows.append({"split": split_label, "model": label, "seed": seed, "images": len(t["targets"]),
                             "errors": len(wrong), "cross_crop_errors": cross, "within_crop_errors": len(wrong) - cross})
            rows.append({"split": split_label, "model": label, "seed": "mean", "images": len(runs[(model, 11)][split]["targets"]),
                         "errors": sum(x[0] for x in per_seed) / 5, "cross_crop_errors": sum(x[1] for x in per_seed) / 5,
                         "within_crop_errors": sum(x[2] for x in per_seed) / 5})
    with open(out / "crop_errors_five_seeds.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["seed"] == "mean":
            print(f"{r['split']:<14} {r['model']:<12} errors {r['errors']:.1f} | cross-crop {r['cross_crop_errors']:.1f} | within-crop {r['within_crop_errors']:.1f}")


if __name__ == "__main__":
    main()
