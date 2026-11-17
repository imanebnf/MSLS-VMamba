import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(open(ROOT / "runs/seed11_layercam/paired_layercam_per_image.csv", encoding="utf-8")))
n = len(rows)
assert n == 350


def mean(key, scale=1.0):
    return scale * sum(float(r[key]) for r in rows) / n


ig = list(csv.DictReader(open(ROOT / "runs/seed11_ig_efficiency/integrated_gradients_summary.csv", encoding="utf-8")))
assert len(ig) == 1 and int(ig[0]["images"]) == 350
ig = ig[0]
table = [
    ("LayerCAM share in lesion, E (%)", mean("baseline_lesion_energy", 100), mean("current_lesion_energy", 100)),
    ("LayerCAM concentration, K", mean("baseline_concentration"), mean("current_concentration")),
    ("LayerCAM peak in lesion (%)", mean("baseline_pointing_hit", 100), mean("current_pointing_hit", 100)),
    ("IG concentration, K", float(ig["baseline_concentration_mean"]), float(ig["current_concentration_mean"])),
    ("Deletion drop, top 10% pixels", float(ig["baseline_top10_deletion_drop_mean"]), float(ig["current_top10_deletion_drop_mean"])),
]
higher_layercam = 100 * sum(float(r["delta_concentration"]) > 0 for r in rows) / n
with open(OUT / "table7_attribution.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["measure", "vmamba", "msls_vmamba", "difference"])
    for name, a, b in table:
        w.writerow([name, a, b, b - a])
    w.writerow(["Images with higher LayerCAM concentration for MSLS-VMamba (%)", "", higher_layercam, ""])
    w.writerow(["Images with higher IG concentration for MSLS-VMamba (%)", "", float(ig["percent_current_more_concentrated"]), ""])
for name, a, b in table:
    print(f"{name:<34} {a:8.3f} {b:8.3f} {b - a:+8.3f}")
print(f"higher concentration for MSLS-VMamba: LayerCAM {higher_layercam:.1f}% of images, IG {float(ig['percent_current_more_concentrated']):.1f}%")
