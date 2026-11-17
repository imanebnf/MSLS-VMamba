import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_table import DRAW_ORDER, N_BOOT, SEEDS, SPLITS, macro_f1, recompute, write

MSLS, VMAMBA, MSV = "current_full_strong_384", "vmamba_baseline_strong_384", "msvmamba_tiny_384"
LABEL = {MSLS: "MSLS-VMamba", VMAMBA: "VMamba", MSV: "MSVMamba-Tiny"}
EXPECTED_PARAMETERS = {MSLS: 30405907, VMAMBA: 29492368, MSV: 32289616}


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for d in sys.argv[2:]:
        for f in Path(d).glob("*final_locked_summary*.json"):
            j = json.loads(f.read_text(encoding="utf-8"))
            if j.get("name") in LABEL and int(j["seed"]) in SEEDS:
                key = (j["name"], int(j["seed"]))
                assert key not in runs, f"duplicate run {key}"
                assert int(j["parameters"]) == EXPECTED_PARAMETERS[j["name"]], key
                runs[key] = j
    for n in (MSLS, VMAMBA, MSV):
        assert all((n, s) in runs for s in SEEDS), f"missing seeds for {n}"
    n_cls = len(runs[(MSLS, SEEDS[0])]["identity"]["class_names"])
    rows, val = [], {}
    for n in (MSLS, VMAMBA, MSV):
        for s in SEEDS:
            j = runs[(n, s)]
            for sp in SPLITS:
                r = recompute(j[sp])
                assert abs(r["macro_f1"] - j[sp]["macro_f1"]) < 1e-6, (n, s, sp)
                val[(n, s, sp)] = r
                rows.append({"run": n, "label": LABEL[n], "seed": s, "split": sp, "best_epoch": j["best_epoch"],
                             "parameters": j["parameters"], **r})
    write(out / "msvmamba_per_seed.csv", rows)
    write(out / "msvmamba_mean_sd.csv", [{"run": n, "label": LABEL[n], "split": sp, "metric": k,
                                     "mean": st.mean(xs := [val[(n, s, sp)][k] for s in SEEDS]), "sample_sd": st.stdev(xs)}
                                    for n in (MSLS, VMAMBA, MSV) for sp in SPLITS
                                    for k in ("accuracy", "balanced_accuracy", "macro_f1")])

    rng = np.random.default_rng(20261031)
    draws = {}
    for sp in DRAW_ORDER:
        n_img = len(runs[(MSLS, SEEDS[0])][sp]["targets"])
        draws[sp] = [(rng.integers(0, n_img, n_img), rng.choice(SEEDS, size=5, replace=True)) for _ in range(N_BOOT)]

    results = []
    for left, right in [(MSLS, VMAMBA), (MSLS, MSV), (MSV, VMAMBA)]:
        for sp in SPLITS:
            y = np.asarray(runs[(left, SEEDS[0])][sp]["targets"])
            for s in SEEDS:
                assert runs[(left, s)][sp]["paths"] == runs[(right, s)][sp]["paths"], "image order differs"
                assert np.array_equal(np.asarray(runs[(right, s)][sp]["targets"]), y)
            d = [val[(left, s, sp)]["macro_f1"] - val[(right, s, sp)]["macro_f1"] for s in SEEDS]
            mean, sd = st.mean(d), st.stdev(d)
            se = sd / 5 ** 0.5
            tq = sps.t.ppf(0.975, 4)
            pl = {s: np.asarray(runs[(left, s)][sp]["predictions"]) for s in SEEDS}
            pr = {s: np.asarray(runs[(right, s)][sp]["predictions"]) for s in SEEDS}
            boot = np.empty(N_BOOT)
            for b, (idx, seeds) in enumerate(draws[sp]):
                fl = {s: macro_f1(y[idx], pl[s][idx], n_cls) for s in set(seeds.tolist())}
                fr = {s: macro_f1(y[idx], pr[s][idx], n_cls) for s in set(seeds.tolist())}
                boot[b] = 100 * float(np.mean([fl[s] - fr[s] for s in seeds]))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            results.append({"comparison": f"{LABEL[left]} minus {LABEL[right]}", "left": left, "right": right, "split": sp,
                            **{f"diff_seed{s}": x for s, x in zip(SEEDS, d)}, "mean_diff": mean, "sd_diff": sd,
                            "t_ci95_low": mean - tq * se, "t_ci95_high": mean + tq * se,
                            "positive_seeds": sum(x > 0 for x in d),
                            "boot_ci95_low": float(lo), "boot_ci95_high": float(hi),
                            "boot_se": float(boot.std(ddof=1))})
    write(out / "msvmamba_paired.csv", results)

    for r in results:
        print(f"{r['comparison']:<34} {SPLITS[r['split']]:<26} {r['mean_diff']:+.2f}  t[{r['t_ci95_low']:+.2f},{r['t_ci95_high']:+.2f}]  "
              f"boot[{r['boot_ci95_low']:+.2f},{r['boot_ci95_high']:+.2f}]  {r['positive_seeds']}/5")


if __name__ == "__main__":
    main()
