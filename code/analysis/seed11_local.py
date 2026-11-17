import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ORDER = ["smallest_25%", "small_mid", "large_mid", "largest_25%"]


def load(sdir, model, seed=11):
    for f in sdir.glob("*__final_locked_summary.json"):
        j = json.loads(f.read_text(encoding="utf-8"))
        if j["model"] == model and int(j["seed"]) == seed:
            return j, f.name.split("__")[0]
    raise SystemExit(f"missing {model} seed {seed}")


def main():
    sdir, extent_csv, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    vm, vm_id = load(sdir, "vmamba_baseline")
    ms, ms_id = load(sdir, "current_full")
    classes = vm["identity"]["class_names"]
    crop = np.array([c.split("_", 1)[0] for c in classes])

    ext = {r["record_id"]: r for r in csv.DictReader(open(extent_csv, encoding="utf-8"))}
    t = vm["plantseg_official_test"]
    assert t["paths"] == ms["plantseg_official_test"]["paths"] and t["targets"] == ms["plantseg_official_test"]["targets"]
    rows = []
    for q in ORDER:
        idx = [i for i, rid in enumerate(t["paths"]) if ext[rid]["mask_extent"] == q]
        y = np.array([t["targets"][i] for i in idx])
        pv = np.array([t["predictions"][i] for i in idx])
        pm = np.array([ms["plantseg_official_test"]["predictions"][i] for i in idx])
        ratios = np.array([float(ext[t["paths"][i]]["mask_ratio"]) for i in idx])
        rows.append({"mask_extent_quartile": q, "images": len(idx), "mask_ratio_min": ratios.min(),
                     "mask_ratio_median": float(np.median(ratios)), "mask_ratio_max": ratios.max(),
                     "vmamba_accuracy": 100 * (pv == y).mean(), "msls_accuracy": 100 * (pm == y).mean(),
                     "delta_accuracy_pp": 100 * ((pm == y).mean() - (pv == y).mean()),
                     "msls_fixes": int(((pm == y) & (pv != y)).sum()), "msls_breaks": int(((pm != y) & (pv == y)).sum()),
                     "classes_present": len(set(y.tolist()))})
    write(out / "seed11_lesion_extent.csv", rows)

    dec = []
    for split in ("plantseg_official_test", "plantdoc_external_clean"):
        y = np.array(vm[split]["targets"])
        pv, pm = np.array(vm[split]["predictions"]), np.array(ms[split]["predictions"])
        ev, em = pv != y, pm != y
        for name, p, e in (("VMamba", pv, ev), ("MSLS-VMamba", pm, em)):
            within = e & (crop[p] == crop[y])
            dec.append({"split": split, "model": name, "images": len(y), "errors": int(e.sum()),
                        "within_crop_errors": int(within.sum()), "cross_crop_errors": int((e & ~within).sum()),
                        "within_crop_share_pct": 100 * within.sum() / max(e.sum(), 1),
                        "shared_errors_both_models": int((ev & em).sum()),
                        "shared_same_wrong_label": int((ev & em & (pv == pm)).sum()),
                        "only_this_model_wrong": int((e & ~(em if name == "VMamba" else ev)).sum())})
    write(out / "seed11_error_decomposition.csv", dec)

    pc = []
    for split in ("plantseg_official_test", "plantdoc_external_clean"):
        y = np.array(vm[split]["targets"])
        fv = f1_score(y, vm[split]["predictions"], labels=range(len(classes)), average=None, zero_division=0)
        fm = f1_score(y, ms[split]["predictions"], labels=range(len(classes)), average=None, zero_division=0)
        for k, c in enumerate(classes):
            pc.append({"split": split, "class": c, "support": int((y == k).sum()), "vmamba_f1": 100 * fv[k],
                       "msls_f1": 100 * fm[k], "delta_pp": 100 * (fm[k] - fv[k])})
    write(out / "seed11_per_class_f1.csv", pc)
    print(f"sources: vmamba {vm_id}, msls {ms_id}")
    for r in rows:
        print(f"{r['mask_extent_quartile']:>13}: n={r['images']} median ratio {r['mask_ratio_median']:.3f} "
              f"VMamba {r['vmamba_accuracy']:.2f} MSLS {r['msls_accuracy']:.2f} delta {r['delta_accuracy_pp']:+.2f} "
              f"(fixes {r['msls_fixes']}, breaks {r['msls_breaks']})")
    for d in dec:
        print(f"{d['split']:>24} {d['model']:>12}: errors {d['errors']}, within-crop {d['within_crop_errors']} "
              f"({d['within_crop_share_pct']:.1f}%), shared {d['shared_errors_both_models']}")


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
