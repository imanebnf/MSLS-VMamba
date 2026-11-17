import csv
import sys
from pathlib import Path

ORDER = ["resnet50", "efficientnet_v2_s", "vit_s16", "vmamba_baseline", "current_full"]
LABEL = {"resnet50": "ResNet50", "efficientnet_v2_s": "EfficientNetV2-S", "vit_s16": "ViT-S/16", "vmamba_baseline": "VMamba",
         "current_full": "MSLS-VMamba"}


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    by = {(r["model"], int(r["batch_size"])): r for r in rows}
    gpus = sorted({r.get("gpu", "") for r in rows})
    table = []
    for m in ORDER:
        b1, b8 = by.get((m, 1)), by.get((m, 8))
        if not b1:
            continue
        table.append({"model": m, "label": LABEL[m], "parameters_M": float(b1["parameters_M"]),
                      "state_dict_MB": float(b1.get("state_dict_MB") or b1["checkpoint_MB"]),
                      "latency_ms_b1": float(b1["latency_ms_per_image"]),
                      "throughput_b8": float(b8["throughput_images_per_second"]) if b8 else None,
                      "latency_ms_per_image_b8": float(b8["latency_ms_per_image"]) if b8 else None,
                      "peak_mem_MB_b1": float(b1["peak_gpu_memory_MB"]), "peak_mem_MB_b8": float(b8["peak_gpu_memory_MB"]) if b8 else None})
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "efficiency_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    f = lambda x, d=2: "–" if x is None else f"{x:.{d}f}"
    md = [f"# Measured inference cost (fp32, 384 × 384, {'; '.join(gpus)})", "",
          "| Model | Params (M) | Weights (MB) | Latency, batch 1 (ms/image) | Throughput, batch 8 (images/s) | Peak memory, batch 1 / 8 (MB) |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in table:
        md.append(f"| {r['label']} | {f(r['parameters_M'])} | {f(r['state_dict_MB'], 1)} | {f(r['latency_ms_b1'])} | {f(r['throughput_b8'], 1)} | "
                  f"{f(r['peak_mem_MB_b1'], 0)} / {f(r['peak_mem_MB_b8'], 0)} |")
    md += ["", "Mean over 60 timed repetitions after 15 warm-up passes, one session, identical procedure for every model. "
           "Per-repetition dispersion was not recorded by the protocol. Latency excludes data loading and host-device transfer. "
           f"Source: `{src.name}`."]
    (out / "efficiency_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[8:]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
