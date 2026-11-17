import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import figure_style as s
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

OUT = Path(sys.argv[1])
SPLITS = [("plantseg_official_test", "plantseg_test", "fig5_cm_plantseg", "PlantSeg test (n = 435)"),
          ("plantdoc_external_clean", "plantdoc", "fig9_cm_plantdoc", "PlantDoc (n = 1,379)")]


def load_runs():
    runs = {}
    for f in glob.glob(str(ROOT / "runs/vmamba_msls_seeds11-55/*final_locked_summary*.json")):
        j = json.loads(Path(f).read_text(encoding="utf-8"))
        runs[(j["model"], int(j["seed"]))] = j
    assert sorted(sd for m, sd in runs if m == "current_full") == [11, 22, 33, 44, 55]
    return runs


def confusion(runs):
    names = runs[("current_full", 11)]["identity"]["class_names"]
    k = len(s.CLASS_ORDER)
    for split, tag, stem, title in SPLITS:
        per, recalls = [], []
        for sd in (11, 22, 33, 44, 55):
            j = runs[("current_full", sd)]
            y, p = np.asarray(j[split]["targets"]), np.asarray(j[split]["predictions"])
            m = np.zeros((k, k))
            order = [names.index(c) for c in s.CLASS_ORDER]
            for a, b in zip(y, p):
                m[order.index(a), order.index(b)] += 1
            per.append(m / m.sum(1, keepdims=True))
            recalls.append([np.mean(p[y == names.index(c)] == names.index(c)) for c in s.CLASS_ORDER])
        share = np.mean(per, axis=0)
        mean = 100 * share
        assert np.allclose(mean.sum(1), 100.0) and np.allclose(np.diag(mean), 100 * np.mean(recalls, axis=0))
        OUT.mkdir(parents=True, exist_ok=True)
        W = s.COL_W
        left, right, bottom, top = 0.50, 0.04, 0.44, 0.24
        side = W - left - right
        H = bottom + side + top
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([left / W, bottom / H, side / W, side / H])
        ax.imshow(mean, cmap=s.SEQ, vmin=0, vmax=100, interpolation="nearest")
        for i in range(k):
            for jx in range(k):
                v = mean[i, jx]
                if i == jx or v >= 10:
                    label = f"{v:.0f}"
                    effects = [pe.withStroke(linewidth=1.6, foreground=s.NAVY)] if len(label) > 2 else None
                    ax.text(jx, i, label, ha="center", va="center", fontsize=s.FS,
                            color="white" if v >= 55 else s.INK, path_effects=effects)
        t = np.arange(k)
        ax.set_xticks(t, [str(i + 1) for i in t])
        ax.set_yticks(t, [str(i + 1) for i in t])
        ax.set_xlabel("Predicted class", labelpad=2)
        ax.set_ylabel("True class", labelpad=2)
        ax.set_title(title, pad=4)
        ax.tick_params(length=0, pad=2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        s.save(fig, OUT, stem)
        plt.close(fig)


def paired():
    d = pd.read_csv(ROOT / "analysis_outputs/main/main_per_seed.csv")
    fig, axes = plt.subplots(1, 2, figsize=(s.COL_W, 2.45), gridspec_kw={"wspace": 0.42, "left": 0.15, "right": 0.99, "top": 0.90, "bottom": 0.17})
    for letter, ax, (split, _, _, title) in zip("ab", axes, SPLITS):
        sub = d[d.split == split]
        v = sub[sub.model == "vmamba_baseline"].set_index("seed").macro_f1
        m = sub[sub.model == "current_full"].set_index("seed").macro_f1
        assert sorted(v.index) == sorted(m.index) == [11, 22, 33, 44, 55] and (m > v).all()
        for sd in v.index:
            ax.plot([0, 1], [v[sd], m[sd]], color=s.LINE, lw=0.6, zorder=1)
        ax.scatter([0] * 5, v.values, s=12, color=s.VMAMBA, zorder=2, linewidths=0)
        ax.scatter([1] * 5, m.values, s=12, color=s.MSLS, zorder=2, linewidths=0)
        ax.scatter([0, 1], [v.mean(), m.mean()], s=42, marker="D", color=[s.VMAMBA, s.MSLS], edgecolor="white", lw=0.6, zorder=3)
        ax.set_xticks([0, 1], ["VMamba", "MSLS-\nVMamba"])
        ax.set_xlim(-0.45, 1.45)
        ax.set_title(f"({letter}) {title.split(' (')[0]}", pad=3)
    axes[0].set_ylabel("Macro-F1 (%)", labelpad=2)
    s.save(fig, OUT, "fig4_paired_seeds")
    plt.close(fig)


def crop_errors(runs):
    names = runs[("vmamba_baseline", 11)]["identity"]["class_names"]
    crop = lambda k: names[k].split("_", 1)[0]
    recount = {}
    for label, model in (("VMamba", "vmamba_baseline"), ("MSLS-VMamba", "current_full")):
        t = runs[(model, 11)]["plantseg_official_test"]
        wrong = [(a, b) for a, b in zip(t["targets"], t["predictions"]) if a != b]
        cross = sum(crop(a) != crop(b) for a, b in wrong)
        recount[label] = (cross, len(wrong) - cross)
    d = pd.read_csv(ROOT / "analysis_outputs/seed11/seed11_error_decomposition.csv")
    d = d[d.split == "plantseg_official_test"].set_index("model")
    counts = {m: (int(d.loc[m, "cross_crop_errors"]), int(d.loc[m, "within_crop_errors"])) for m in ("VMamba", "MSLS-VMamba")}
    assert counts == recount, (counts, recount)
    fig, ax = plt.subplots(figsize=(s.COL_W, 1.95))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.97, bottom=0.135)
    w, x = 0.34, (0.0, 1.0)
    for off, (model, color) in zip((-w / 2, w / 2), (("VMamba", s.VMAMBA), ("MSLS-VMamba", s.MSLS))):
        bars = ax.bar([v + off for v in x], counts[model], width=w - 0.02, color=color, label=model)
        for b, v in zip(bars, counts[model]):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, str(v), ha="center", va="bottom", fontsize=s.FS)
    ax.set_xticks(x, ["Cross-crop errors", "Within-crop errors"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 66)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylabel("Test images", labelpad=2)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper left", handlelength=1.1, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.25)
    s.save(fig, OUT, "fig6_crop_errors")
    plt.close(fig)


def lesion_extent():
    o = pd.read_csv(ROOT / "analysis_outputs/alignment_interventions/layercam_concentration_by_extent_test_five_seeds.csv")
    o = o[o.model.isin(["VMamba", "MSLS-VMamba"])]
    fig, ax = plt.subplots(figsize=(s.COL_W, 2.6))
    fig.subplots_adjust(left=0.135, right=0.99, top=0.985, bottom=0.25)
    labs = list(dict.fromkeys(o.extent))
    xs = np.arange(len(labs))
    for off, name, col in ((-0.12, "VMamba", s.VMAMBA), (0.12, "MSLS-VMamba", s.MSLS)):
        q = o[o.model == name]
        ax.errorbar(xs + off, q["mean"], yerr=[q["mean"] - q["lo"], q["hi"] - q["mean"]], fmt="o", ms=4, color=col,
                    elinewidth=s.LW, capsize=2, capthick=s.LW_THIN, label=name)
    ax.axhline(1.0, color=s.LINE, lw=s.LW_THIN, ls=(0, (4, 3)), zorder=0)
    ns = o[o.model == "VMamba"]["n"].tolist()
    ax.set_xticks(xs, [f"{b}\n(n = {n})" for b, n in zip(labs, ns)])
    ax.set_xlabel("Annotated lesion area (share of image)", labelpad=2)
    ax.set_ylabel("LayerCAM concentration", labelpad=2)
    ax.set_yticks(range(1, 8))
    ax.legend(loc="upper right", borderaxespad=0.1, handletextpad=0.2)
    s.save(fig, OUT, "fig8_lesion_extent")
    plt.close(fig)


if __name__ == "__main__":
    s.apply()
    runs = load_runs()
    confusion(runs)
    paired()
    crop_errors(runs)
    lesion_extent()
