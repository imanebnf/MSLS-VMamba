import csv
import sys
from pathlib import Path

ORDER = ["current_full", "vmamba_baseline", "current_full_no_mask_loss", "current_full_same_class_permuted_masks",
         "vmamba_plain_multitask", "current_full_fine_only", "current_full_mid_only", "current_full_coarse_only"]
LABEL = {"current_full": "MSLS-VMamba (full)", "vmamba_baseline": "VMamba (no lesion head)",
         "current_full_no_mask_loss": "Mask loss disabled", "current_full_same_class_permuted_masks": "Same-class permuted masks",
         "vmamba_plain_multitask": "Plain multitask (masks, no regional pooling)", "current_full_fine_only": "Fine scale only",
         "current_full_mid_only": "Middle scale only", "current_full_coarse_only": "Coarse scale only"}
SPLITS = [("plantseg_official_test", "plantseg_test"), ("plantdoc_external_clean", "plantdoc")]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = {r["model"]: r for r in csv.DictReader(open(src, encoding="utf-8"))}
    ref = rows["current_full"]
    table = []
    for m in [x for x in ORDER if x in rows]:
        r = rows[m]
        row = {"model": m, "label": LABEL[m], "parameters_M": num(r["parameters"]) / 1e6}
        for key, name in SPLITS:
            v = num(r[f"{key}_macro_f1"])
            row[f"{name}_macro_f1"] = v
            row[f"{name}_gain_pp"] = None if m == "current_full" else num(ref[f"{key}_macro_f1"]) - v
        row["plantseg_test_mask_dice"] = num(r["plantseg_official_test_mask_dice"])
        table.append(row)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "seed11_ablation.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    for row in table:
        print(f"{row['label']:<46} PlantSeg test {row['plantseg_test_macro_f1']:.2f}   PlantDoc {row['plantdoc_macro_f1']:.2f}")


if __name__ == "__main__":
    main()
