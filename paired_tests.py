"""
paired_tests.py  --  the statistical testing your guide asked for.

Runs on data you ALREADY HAVE. No retraining.

WHAT IT COMPUTES
  Over the 50 held-out welds of the Protocol B test set, it treats each weld
  as one paired observation (baseline accuracy on that weld vs hybrid
  accuracy on that weld) and runs:

    1. Paired t-test            (parametric, assumes roughly normal diffs)
    2. Wilcoxon signed-rank     (non-parametric, no normality assumption)
    3. Exact McNemar            (tile-level, paired 2x2)
    4. Weld-clustered bootstrap (95% CI on the accuracy difference)
    5. Cohen's d_z              (effect size for the paired design)

  The weld is the right unit of analysis: tiles cut from one radiograph are
  not independent, so a test over 2,194 tiles would overstate significance.

  If you later run cross-validation, point --folds at the folder of fold
  metrics and it will ALSO run the paired t-test and Wilcoxon across folds,
  which is the form your guide asked for.

OUTPUT
  stats_table.tex     a ready-to-paste LaTeX table
  stats_results.json  all values

USAGE
    python paired_tests.py
    python paired_tests.py --folds cv_results      # after cross-validation
"""

import os, re, json, argparse
from math import comb
import numpy as np
import pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="baseline_results_weldlevel")
ap.add_argument("--hyb",  default="hybrid_results_weldlevel")
ap.add_argument("--folds", default=None,
                help="folder with fold_*_baseline.json / fold_*_hybrid.json")
ap.add_argument("--boot", type=int, default=10000)
args = ap.parse_args()

R = {}

# ------------------------------------------------ load paired predictions
pb = pd.read_csv(os.path.join(args.base, "test_predictions.csv"))
ph = pd.read_csv(os.path.join(args.hyb,  "test_predictions.csv"))
df = pb.merge(ph, on=["filename", "true_class"], suffixes=("_b", "_h"))
if len(df) != len(pb):
    raise SystemExit("prediction files do not align; same split?")
df["ok_b"] = df.predicted_class_b == df.true_class
df["ok_h"] = df.predicted_class_h == df.true_class

PARENT = re.compile(r"_\[\d+\]\[\d+\].*\.png$", re.IGNORECASE)
df["weld"] = df.filename.map(lambda f: PARENT.sub("", os.path.basename(f)))

g = df.groupby("weld").agg(n=("ok_b", "size"), cb=("ok_b", "sum"),
                           ch=("ok_h", "sum"))
acc_b = (g.cb / g.n).values
acc_h = (g.ch / g.n).values
d = acc_h - acc_b
W = len(g)

R["n_test_tiles"] = int(len(df))
R["n_test_welds"] = int(W)
R["overall_acc_baseline"] = float(df.ok_b.mean())
R["overall_acc_hybrid"] = float(df.ok_h.mean())
R["mean_weld_acc_baseline"] = float(acc_b.mean())
R["mean_weld_acc_hybrid"] = float(acc_h.mean())

# ---------------------------------------------------- 1. paired t-test
t_stat, t_p = stats.ttest_rel(acc_h, acc_b)
R["paired_t_statistic"] = float(t_stat)
R["paired_t_p"] = float(t_p)
R["paired_t_df"] = int(W - 1)

# ------------------------------------------- 2. Wilcoxon signed-rank
nz = int((d != 0).sum())
if nz:
    w_stat, w_p = stats.wilcoxon(acc_h, acc_b, zero_method="wilcox",
                                 alternative="two-sided")
    R["wilcoxon_statistic"] = float(w_stat)
    R["wilcoxon_p"] = float(w_p)
else:
    R["wilcoxon_statistic"] = None; R["wilcoxon_p"] = 1.0
R["wilcoxon_nonzero_pairs"] = nz

# ---------------------------------------------------- 3. exact McNemar
b01 = int((df.ok_b & ~df.ok_h).sum())
b10 = int((~df.ok_b & df.ok_h).sum())
m = b01 + b10
p_mcn = min(1.0, 2 * sum(comb(m, i) for i in range(min(b01, b10) + 1)) / 2**m) if m else 1.0
R["mcnemar_baseline_right_hybrid_wrong"] = b01
R["mcnemar_baseline_wrong_hybrid_right"] = b10
R["mcnemar_p"] = float(p_mcn)

# ------------------------------------------ 4. weld-clustered bootstrap
rng = np.random.default_rng(42)
N, CB, CH = g.n.values, g.cb.values, g.ch.values
diffs = np.empty(args.boot)
for i in range(args.boot):
    k = rng.integers(0, W, W)
    diffs[i] = (CH[k].sum() - CB[k].sum()) / N[k].sum()
lo, hi = np.percentile(diffs, [2.5, 97.5])
R["bootstrap_diff_pp"] = float(100 * (CH.sum() - CB.sum()) / N.sum())
R["bootstrap_ci_low_pp"] = float(100 * lo)
R["bootstrap_ci_high_pp"] = float(100 * hi)
R["bootstrap_reps"] = args.boot

# ------------------------------------------------- 5. effect size d_z
R["cohens_dz"] = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else None
R["welds_hybrid_better"] = int((d > 0).sum())
R["welds_baseline_better"] = int((d < 0).sum())
R["welds_tied"] = int((d == 0).sum())

# ------------------------------------------ optional: across CV folds
fold_rows = []
if args.folds and os.path.isdir(args.folds):
    for f in sorted(os.listdir(args.folds)):
        mo = re.match(r"fold_(\d+)_(baseline|hybrid)\.json$", f)
        if mo:
            j = json.load(open(os.path.join(args.folds, f)))
            fold_rows.append((int(mo.group(1)), mo.group(2), j["accuracy"]))
if fold_rows:
    fdf = pd.DataFrame(fold_rows, columns=["fold", "model", "acc"]) \
            .pivot(index="fold", columns="model", values="acc").dropna()
    fb, fh = fdf["baseline"].values, fdf["hybrid"].values
    ft, fp = stats.ttest_rel(fh, fb)
    R["cv_folds"] = int(len(fdf))
    R["cv_mean_baseline"] = float(fb.mean()); R["cv_std_baseline"] = float(fb.std(ddof=1))
    R["cv_mean_hybrid"] = float(fh.mean());  R["cv_std_hybrid"] = float(fh.std(ddof=1))
    R["cv_paired_t_p"] = float(fp); R["cv_paired_t_statistic"] = float(ft)
    if (fh != fb).any():
        wS, wP = stats.wilcoxon(fh, fb, zero_method="wilcox")
        R["cv_wilcoxon_p"] = float(wP); R["cv_wilcoxon_statistic"] = float(wS)
    print(f"\nCross-validation across {len(fdf)} folds included.")

# ---------------------------------------------------------- report
def fmt_p(p):
    return f"$<10^{{-4}}$" if p < 1e-4 else (f"${p:.2e}$".replace("e-0", r"\times10^{-") + "}$"
            if p < 1e-3 else f"{p:.4f}")

print("=" * 66)
print(f"Unit of analysis: the weld.  {W} held-out welds, {len(df):,} tiles.")
print(f"  baseline accuracy {100*R['overall_acc_baseline']:.2f}%   "
      f"hybrid {100*R['overall_acc_hybrid']:.2f}%")
print("-" * 66)
print(f"Paired t-test        t({W-1}) = {t_stat:.3f},  p = {t_p:.4g}")
print(f"Wilcoxon signed-rank W = {R['wilcoxon_statistic']},  p = {R['wilcoxon_p']:.4g}"
      f"   ({nz} welds differ, {R['welds_tied']} tied)")
print(f"Exact McNemar        {b10} vs {b01} discordant tiles,  p = {p_mcn:.4g}")
print(f"Bootstrap 95% CI     [{R['bootstrap_ci_low_pp']:.2f}, "
      f"{R['bootstrap_ci_high_pp']:.2f}] pp")
print(f"Cohen's d_z          {R['cohens_dz']:.3f}"
      if R["cohens_dz"] is not None else "Cohen's d_z          n/a")
print("=" * 66)

sig = [n for n, p in [("paired t-test", t_p), ("Wilcoxon", R["wilcoxon_p"]),
                      ("McNemar", p_mcn)] if p < 0.05]
print(("Significant at 0.05 by: " + ", ".join(sig)) if sig
      else "No test reaches p < 0.05.")
if nz < 6:
    print("WARNING: Wilcoxon with fewer than 6 differing pairs cannot reach "
          "p<0.05 regardless of the data.")

json.dump(R, open("stats_results.json", "w"), indent=2)

with open("stats_table.tex", "w") as f:
    f.write(r"""\begin{table}[!t]
    \centering
    \caption{Statistical comparison of the two models on the Protocol B test
    set. The weld is the unit of analysis for the paired tests, since tiles
    cropped from one radiograph are not independent.}
    \label{tab:stats}
    \setlength{\tabcolsep}{4pt}
    \begin{tabular}{@{}lcc@{}}
        \toprule
        \textbf{Test} & \textbf{Statistic} & \textbf{$p$-value} \\
        \midrule
""")
    f.write(f"        Paired $t$-test (\\textit{{n}}~$=${W} welds) & "
            f"$t({W-1})={t_stat:.3f}$ & {fmt_p(t_p)} \\\\\n")
    f.write(f"        Wilcoxon signed-rank & $W={R['wilcoxon_statistic']:.1f}$ & "
            f"{fmt_p(R['wilcoxon_p'])} \\\\\n")
    f.write(f"        Exact McNemar (tiles) & ${b10}$ vs ${b01}$ & "
            f"{fmt_p(p_mcn)} \\\\\n")
    f.write(r"        \midrule" + "\n")
    f.write(f"        Accuracy difference & {R['bootstrap_diff_pp']:.2f} pp & --- \\\\\n")
    f.write(f"        Bootstrap 95\\% CI & [{R['bootstrap_ci_low_pp']:.2f}, "
            f"{R['bootstrap_ci_high_pp']:.2f}] pp & --- \\\\\n")
    if R["cohens_dz"] is not None:
        f.write(f"        Cohen's $d_z$ & {R['cohens_dz']:.3f} & --- \\\\\n")
    f.write(r"""        \bottomrule
    \end{tabular}
\end{table}
""")
# ---- figure: per-weld accuracy difference ----------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.linewidth": 0.6, "savefig.dpi": 400,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
order = np.argsort(d)
dd = 100 * d[order]
sizes = N[order]
fig, ax = plt.subplots(figsize=(3.45, 2.0))
colors = ["#c0392b" if v < 0 else ("#bbbbbb" if v == 0 else "#2c6fbb") for v in dd]
ax.bar(np.arange(W), dd, color=colors, width=0.9)
ax.axhline(0, color="0.3", lw=0.7)
ax.set_xlabel(f"Held-out welds, sorted ({W} total)")
ax.set_ylabel("Hybrid $-$ baseline\naccuracy on that weld (pp)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#2c6fbb", label=f"hybrid better ({int((d>0).sum())})"),
                   Patch(color="#bbbbbb", label=f"tied ({int((d==0).sum())})"),
                   Patch(color="#c0392b", label=f"baseline better ({int((d<0).sum())})")],
          frameon=False, loc="upper left")
fig.savefig("wl_fig05_weld_diff.png"); plt.close(fig)
print("wrote wl_fig05_weld_diff.png")

# ---- LaTeX macros the paper pulls in automatically --------------------
def tex_p(p):
    if p < 1e-4: return r"$<10^{-4}$"
    if p < 1e-3: return "$" + f"{p:.2e}".replace("e-0", r"\times 10^{-").replace("e-", r"\times 10^{-") + "}$"
    return f"{p:.4f}"

M = {
 "NWelds": str(W), "NDiffer": str(nz), "NTied": str(R["welds_tied"]),
 "WeldsHbet": str(R["welds_hybrid_better"]), "WeldsBbet": str(R["welds_baseline_better"]),
 "PairedT": f"{t_stat:.3f}", "PairedTdf": str(W-1), "PairedTP": tex_p(t_p),
 "WilcoxW": (f"{R['wilcoxon_statistic']:.1f}" if R["wilcoxon_statistic"] is not None else "--"),
 "WilcoxP": tex_p(R["wilcoxon_p"]),
 "CohenDz": (f"{R['cohens_dz']:.3f}" if R["cohens_dz"] is not None else "--"),
 "McnA": str(b10), "McnB": str(b01), "McnPv": tex_p(p_mcn),
 "DiffPP": f"{R['bootstrap_diff_pp']:.2f}",
 "CIlo": f"{R['bootstrap_ci_low_pp']:.2f}", "CIhi": f"{R['bootstrap_ci_high_pp']:.2f}",
}
if "cv_folds" in R:
    M.update({
      "CVfolds": str(R["cv_folds"]),
      "CVbaseMean": f"{100*R['cv_mean_baseline']:.2f}",
      "CVbaseSD":   f"{100*R['cv_std_baseline']:.2f}",
      "CVhybMean":  f"{100*R['cv_mean_hybrid']:.2f}",
      "CVhybSD":    f"{100*R['cv_std_hybrid']:.2f}",
      "CVtP":       tex_p(R["cv_paired_t_p"]),
      "CVwP":       tex_p(R.get("cv_wilcoxon_p", 1.0)),
    })
with open("stats_macros.tex", "w") as f:
    f.write("% Auto-generated by paired_tests.py -- do not edit by hand.\n")
    for k, v in sorted(M.items()):
        f.write(f"\\renewcommand{{\\{k}}}{{{v}}}\n")

print("\nwrote stats_table.tex, stats_results.json and stats_macros.tex")
print("Upload stats_macros.tex to Overleaf next to the .tex.")
