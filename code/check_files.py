import csv
import glob
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
while not (os.path.isdir(os.path.join(ROOT, "data")) and os.path.isdir(os.path.join(ROOT, "runs"))):
    if os.path.dirname(ROOT) == ROOT:
        sys.exit("package root (folder holding data/ and runs/) not found")
    ROOT = os.path.dirname(ROOT)
WRITE = "--write" in sys.argv
FAIL = []


def p(*parts):
    return os.path.join(ROOT, *parts)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(ok, message):
    print(("OK   " if ok else "FAIL ") + message)
    if not ok:
        FAIL.append(message)


def rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fingerprints():
    for row in rows(p("data/manifests/manifest_fingerprints.csv")):
        path = p("data/manifests", row["manifest"])
        ok = os.path.exists(path) and sha256(path) == row["sha256"]
        n = len(rows(path)) if os.path.exists(path) else -1
        check(ok and n == int(row["rows"]), f"{row['manifest']}: SHA-256 and {row['rows']} rows match the locked fingerprint")


def class_mapping():
    names = json.load(open(p("data/cleaning_protocol.json"), encoding="utf-8"))["class_names"]
    paper = ["Apple scab", "Apple rust", "Corn gray leaf spot", "Corn rust", "Corn northern leaf blight", "Grape black rot",
             "Potato early blight", "Potato late blight", "Squash powdery mildew", "Tomato bacterial spot",
             "Tomato early blight", "Tomato late blight", "Tomato leaf mold", "Tomato Septoria leaf spot",
             "Tomato mosaic virus", "Tomato yellow leaf curl"]
    originals = defaultdict(set)
    sources = glob.glob(p("data/manifests/*.csv")) + [p("data/exclusions/exclusions.csv")]
    for path in sources:
        if path.endswith("manifest_fingerprints.csv"):
            continue
        for row in rows(path):
            originals[(row["harmonized_class"], row["dataset"])].add(row["original_class"].strip())
    assert set(c for c, _ in originals) == set(names), "unexpected harmonized class"
    table = [{"class_index": i, "harmonized_class": c, "paper_label": paper[i],
              "plantseg_original_class": " || ".join(sorted(originals[(c, "PlantSeg")])),
              "plantdoc_original_class": " || ".join(sorted(originals[(c, "PlantDoc")]))} for i, c in enumerate(names)]
    out = p("data/class_mapping.csv")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(table[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(table)
    if WRITE:
        open(out, "w", encoding="utf-8", newline="").write(buf.getvalue())
    check(os.path.exists(out) and open(out, encoding="utf-8", newline="").read() == buf.getvalue(),
          "class_mapping.csv matches the manifests and the exclusion list (16 classes, one-to-one)")
    one_to_one = all(len(originals[(c, d)]) == 1 for c in names for d in ("PlantSeg", "PlantDoc"))
    check(one_to_one, "each harmonized class has exactly one original class name per dataset (trailing spaces ignored)")


def table_s1():
    ex = rows(p("data/exclusions/exclusions.csv"))
    by_part = Counter((r["dataset"], r["source_split"], r["exclusion_reason"]) for r in ex)
    s1 = {(r["dataset"], r["source_split"], r["exclusion_reason"]): int(r["images"])
          for r in rows(p("supplementary_tables/S1_exclusions_by_partition_reason.csv"))}
    check(dict(by_part) == s1, f"S1_exclusions_by_partition_reason.csv recounted from exclusions.csv ({len(ex)} records)")
    with open(p("supplementary_tables/S1_exclusions_by_class_reason.csv"), newline="", encoding="utf-8") as fh:
        grid = list(csv.reader(fh))
    cols = list(zip(grid[0][1:], grid[1][1:]))
    s1b = {(r[0], d, reason): int(v) for r in grid[3:] for (d, reason), v in zip(cols, r[1:])}
    by_class = Counter((r["harmonized_class"], r["dataset"], r["exclusion_reason"]) for r in ex)
    check(all(by_class.get(k, 0) == v for k, v in s1b.items()) and sum(s1b.values()) == len(ex),
          "S1_exclusions_by_class_reason.csv recounted from exclusions.csv")


def corrected_masks():
    test = rows(p("data/manifests/plantseg_official_test.csv"))
    ref = rows(p("data/plantseg_test_corrected_mask_references.csv"))
    ok = [r["record_id"] for r in ref] == [r["record_id"] for r in test]
    rule = all(r["correct_mask_reference"] == re.sub(r"images/test/(.*)\.jpg$", r"annotations/test/\1.png", t["image_reference"])
               for r, t in zip(ref, test))
    distinct_locked = len({t["mask_reference"] for t in test})
    check(ok and rule, f"corrected test-mask references: 435 records, own annotation each (locked manifest has {distinct_locked} distinct mask reference(s))")


def run_index():
    out_rows = []
    for path in sorted(glob.glob(p("runs", "*", "*__final_locked_summary.json")) + glob.glob(p("runs", "*", "*__training_summary.json"))):
        j = json.load(open(path, encoding="utf-8"))
        spec, cfg = j.get("spec", {}), j.get("config", {})
        payload = j.get("identity", {})
        pk = (j.get("runtime") or {}).get("packages", {})
        out_rows.append({
            "run_group": os.path.basename(os.path.dirname(path)),
            "run_name": j["name"], "model": j["model"], "seed": j["seed"], "parameters": j.get("parameters"),
            "best_epoch": j.get("best_epoch"), "epochs_completed": j.get("epochs_completed", ""),
            "checkpoint_sha256": j.get("checkpoint_sha256", "not recorded in the saved summary"),
            "protocol_fingerprint": j.get("protocol_fingerprint", ""),
            "data_content_fingerprint": payload.get("data_content_fingerprint", ""),
            "pretrained_snapshot_sha256": payload.get("pretrained_snapshot_sha256", ""),
            "img_size": spec.get("img_size"), "batch_size": spec.get("batch_size"),
            "accumulation_steps": spec.get("accumulation_steps"), "encoder_lr": spec.get("encoder_lr"),
            "head_lr": spec.get("head_lr"), "warmup_epochs": spec.get("warmup_epochs"),
            "label_smoothing": spec.get("label_smoothing"),
            "lambda_mask": spec.get("lambda_mask", cfg.get("lambda_mask", "")),
            "mask_target_mode": spec.get("mask_target_mode", ""),
            "epochs": cfg.get("epochs", ""), "weight_decay": cfg.get("weight_decay", ""), "grad_clip": cfg.get("grad_clip", ""),
            "exif_policy": cfg.get("exif_policy", ""),
            "torch": pk.get("torch", ""), "mamba_ssm": pk.get("mamba-ssm", ""), "timm": pk.get("timm", ""),
            "transformers": pk.get("transformers", ""), "gpu": (j.get("runtime") or {}).get("gpu", ""),
            "evaluated_utc": j.get("evaluated_utc", ""),
            "test_sets_in_file": bool(j.get("official_tests_evaluated")) and "plantseg_official_test" in j,
            "summary_file": os.path.relpath(path, ROOT).replace(os.sep, "/")})
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(out_rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(out_rows)
    out = p("runs/run_index.csv")
    if WRITE:
        open(out, "w", encoding="utf-8", newline="").write(buf.getvalue())
    check(os.path.exists(out) and open(out, encoding="utf-8", newline="").read() == buf.getvalue(),
          f"runs/run_index.csv matches the {len(out_rows)} run files")


def manifest():
    listed = rows(p("MANIFEST.csv"))
    bad = [r["path"] for r in listed if not os.path.exists(p(r["path"])) or sha256(p(r["path"])) != r["sha256"]
           or os.path.getsize(p(r["path"])) != int(r["size_bytes"])]
    check(not bad, f"MANIFEST.csv: {len(listed) - len(bad)}/{len(listed)} files match size and SHA-256" + (f"; mismatches: {bad[:5]}" if bad else ""))


if __name__ == "__main__":
    fingerprints()
    class_mapping()
    table_s1()
    corrected_masks()
    run_index()
    if not WRITE:
        manifest()
    print("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} CHECK(S) FAILED")
    sys.exit(1 if FAIL else 0)
