import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text

for f in glob.glob("/usr/share/fonts/truetype/liberation*/LiberationSerif-*.ttf"):
    fm.fontManager.addfont(f)

NAVY = "#1F3A5F"
LINE, INK = "#6B6F73", "#1A1A1A"
VMAMBA, MSLS = "#8A8A8A", "#1F3A5F"
FULL_W, COL_W = 6.77, 3.23
FS = 10.0
LW, LW_THIN = 1.0, 0.75
SEQ = LinearSegmentedColormap.from_list("navy_seq", ["#FFFFFF", "#C7D0DC", "#7F93AB", "#1F3A5F"])
RC = {"font.family": "Liberation Serif", "font.size": FS, "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none",
      "axes.linewidth": LW_THIN, "axes.edgecolor": LINE, "axes.labelcolor": INK, "axes.labelsize": FS, "axes.titlesize": FS,
      "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": FS, "ytick.labelsize": FS, "xtick.major.width": LW_THIN,
      "ytick.major.width": LW_THIN, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
      "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": FS, "lines.linewidth": LW, "text.color": INK}

CLASS_ORDER = ["apple_scab", "apple_rust", "corn_gray_leaf_spot", "corn_rust", "corn_northern_blight", "grape_black_rot",
               "potato_early_blight", "potato_late_blight", "squash_powdery_mildew", "tomato_bacterial_spot",
               "tomato_early_blight", "tomato_late_blight", "tomato_leaf_mold", "tomato_septoria", "tomato_mosaic_virus",
               "tomato_yellow_curl"]


def apply():
    matplotlib.rcParams.update(RC)


def check_layout(fig, min_size=FS, pad=0.003):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches()
    d = fig.dpi
    items, out = [], []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if t.get_fontsize() < min_size - 1e-6:
            out.append(f"font {t.get_fontsize():.1f}: {t.get_text()!r}")
        b = t.get_window_extent(r)
        x0, y0, x1, y1 = b.x0 / d, b.y0 / d, b.x1 / d, b.y1 / d
        if x0 < -1e-3 or y0 < -1e-3 or x1 > W + 1e-3 or y1 > H + 1e-3:
            out.append(f"clipped: {t.get_text()!r}")
        items.append((t.get_text(), x0 + pad, y0 + pad, x1 - pad, y1 - pad))
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if a[1] < b[3] and b[1] < a[3] and a[2] < b[4] and b[2] < a[4]:
                out.append(f"overlap: {a[0]!r} / {b[0]!r}")
    return out


def save(fig, out_dir, stem):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = check_layout(fig)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=600)
    print(f"{stem}: {fig.get_size_inches()[0] * 2.54:.1f} x {fig.get_size_inches()[1] * 2.54:.1f} cm, layout problems: {problems}")
