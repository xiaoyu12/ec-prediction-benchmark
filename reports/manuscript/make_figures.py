#!/usr/bin/env python
"""Draw the manuscript figures (PDF) from the JSON files in ``data/``.

    python3 scripts/manuscript_data.py                       # -> data/results.json, data/calibration.json
    python3 scripts/identity_bins.py ... --out reports/manuscript/data/identity_bins.json
    python3 scripts/category_bootstrap.py ... --out reports/manuscript/data/bootstrap_vs_homology.json
    envs/esmc/bin/python reports/manuscript/make_figures.py  # needs matplotlib

``data/rarity_bins.json`` was transcribed from ``scripts/rare_eval.py`` (its ``source`` field has the
command). Every number in the figures comes from these files.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
INK, QUIET, GRID = "#111827", "#6b7280", "#e5e7eb"
CAT = {  # Okabe-Ito colours, colour-blind safe
    "Homology only": "#D55E00",
    "Sequence PLM": "#0072B2",
    "PLM + SAE features": "#56B4E9",
    "Structure-aware PLM": "#009E73",
    "Fine-tuned PLM": "#CC79A7",
    "PLM combination": "#E69F00",
    "PLM + homology": "#000000",
    "Published method": "#999999",
}
TESTS = ["new", "price", "uniprot-all", "uniprot-multi", "test70", "test50"]
SET_LABEL = {"new": "New-392", "price": "Price-149", "uniprot-all": "UniProt-all", "uniprot-multi": "UniProt-multi",
             "test70": "Test-70", "test50": "Test-50", "test_mean": "Test mean"}
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "axes.titlesize": 8.5,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})

rows = json.loads((DATA / "results.json").read_text())
by_name = {r["name"]: r for r in rows}
HOM = by_name["MMseqs2 + Foldseek-AFDB, best hit"]  # the id50-selected homology-only baseline


# ---------------------------------------------------------------- Figure 1: study design
def box(ax, x, y, w, h, text, fc, ec=INK, fs=6.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, a, b, color=QUIET):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=color, lw=0.9,
                                 shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(7.4, 4.3))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
box(ax, 0.01, 0.55, 0.12, 0.16, "Query\nsequence", "#f3f4f6", bold=True)
box(ax, 0.17, 0.80, 0.29, 0.14, "Homology search\nMMseqs2 (sequence)\nFoldseek (3Di + sequence)", "#fde7da")
box(ax, 0.51, 0.80, 0.18, 0.14, "Top-hit transfer\n$p_{hom}$, identity", "#fde7da")
box(ax, 0.17, 0.55, 0.29, 0.16, "3Di structure tokens\nAlphaFold DB models (pLDDT)\nor ProstT5 prediction", "#dcf1ea")
box(ax, 0.17, 0.24, 0.29, 0.22, "Protein language models\nESM2 650M/3B, ESMC 600M/6B,\nProtT5;  SAE features (ESMC);\nstructure-aware: ProstT5, SaProt",
    "#dbeaf7")
box(ax, 0.51, 0.28, 0.18, 0.14, "Pooled embeddings\n→ MLP heads\n→ ensemble  $p_{model}$", "#dbeaf7")
box(ax, 0.75, 0.52, 0.23, 0.22, "Identity gate\n$p=(1-a)\\,p_{model}+a\\,p_{hom}$\n$a$ grows with top-hit\nidentity", "#f3f4f6")
box(ax, 0.77, 0.25, 0.19, 0.14, "EC calls\n$p\\geq0.5$,\n1–5 per enzyme", "#ffffff")
arrow(ax, (0.13, 0.66), (0.17, 0.86))
arrow(ax, (0.13, 0.63), (0.17, 0.63))
arrow(ax, (0.13, 0.60), (0.17, 0.36))
arrow(ax, (0.46, 0.87), (0.51, 0.87))
arrow(ax, (0.315, 0.71), (0.315, 0.80))   # 3Di feeds Foldseek
arrow(ax, (0.315, 0.55), (0.315, 0.46))   # 3Di feeds SaProt
arrow(ax, (0.46, 0.35), (0.51, 0.35))
arrow(ax, (0.69, 0.87), (0.86, 0.74))
arrow(ax, (0.69, 0.35), (0.80, 0.52))
arrow(ax, (0.865, 0.52), (0.865, 0.39))
ax.text(0.60, 0.965, "homology-only baseline: transfer alone", ha="center", fontsize=7, color=CAT["Homology only"])
ax.text(0.60, 0.215, "PLM / structure-only: heads alone", ha="center", fontsize=7, color=CAT["Sequence PLM"])
y = 0.02
steps = ["Train: split100\n227,360 enzymes; 5,242 ECs",
         "Select on id50 fold\n10,422 enzymes; ≤50% id.",
         "Retrain chosen model\non all of split100",
         "Score six test sets once\n3,394 enzymes"]
for i, t in enumerate(steps):
    box(ax, 0.01 + i * 0.25, y, 0.22, 0.13, t, "#f3f4f6", fs=6.4)
    if i:
        arrow(ax, (i * 0.25 - 0.015, y + 0.065), (0.01 + i * 0.25, y + 0.065))
fig.savefig(HERE / "fig1_design.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: best of each category
order = list(CAT)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
for ax, key, title, lim, flip in [(axes[0], "id50", "a  Validation (id50 fold)", (0.855, 0.945), 0.911),
                                  (axes[1], "test_mean", "b  Mean of six test sets", (0.495, 0.665), 0.6156)]:
    for i, cat in enumerate(order):
        members = [r for r in rows if r["category"] == cat and r[key] is not None]
        if not members:
            continue
        xs = [r[key] for r in members]
        ax.scatter(xs, [i] * len(xs), s=16, facecolor="none", edgecolor=CAT[cat], lw=0.8, alpha=0.7, zorder=2)
        best = max(members, key=lambda r: r[key])
        ax.scatter([best[key]], [i], s=46, color=CAT[cat], zorder=3)
        ax.text(best[key], i - 0.36, f"{best[key]:.3f}  {best['name']}", fontsize=6.2, color=INK, va="center",
                ha="right" if best[key] > flip else "left")
    ax.axvline(HOM[key], color=CAT["Homology only"], ls=(0, (4, 3)), lw=1)
    homs = [r[key] for r in rows if r["category"] == "Homology only"]
    ax.axvspan(min(homs), max(homs), color=CAT["Homology only"], alpha=0.08, lw=0)
    ax.set_xlim(*lim)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title(title)
    ax.set_xlabel("Per-enzyme F1")
axes[0].set_yticks(range(len(order)), order)
axes[0].set_ylim(len(order) - 0.5, -0.8)
for lab, cat in zip(axes[0].get_yticklabels(), order):
    lab.set_color(CAT[cat] if cat != "Published method" else QUIET)
axes[0].tick_params(axis="y", length=0)
fig.tight_layout()
fig.savefig(HERE / "fig2_categories.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3: id50 vs test mean
fig, ax = plt.subplots(figsize=(5.2, 3.9))
pts = [r for r in rows if r["id50"] is not None]
for cat in order:
    sel = [r for r in pts if r["category"] == cat]
    if sel:
        ax.scatter([r["id50"] for r in sel], [r["test_mean"] for r in sel], s=26, color=CAT[cat], label=cat,
                   edgecolor="white", lw=0.5, zorder=3)
ax.axhline(HOM["test_mean"], color=CAT["Homology only"], ls=(0, (4, 3)), lw=0.9)
ax.axvline(HOM["id50"], color=CAT["Homology only"], ls=(0, (4, 3)), lw=0.9)
labels = {"Gradient-boosted re-ranker": (-4, -9), "ESMC-600M, fine-tuned": (4, -9), "ESM2-3B": (-4, 6),
          "Six-head ensemble + homology": (-6, 5), "c3 + homology": (4, -9), "SaProt-650M, pLDDT-masked": (4, -2),
          "MMseqs2 + Foldseek-AFDB, best hit": (5, -7), "ESMC-600M": (4, -6), "ESMC-6B": (4, -2),
          "MIL-BC+ (released)": (4, -2), "ProstT5": (4, -6)}
for r in pts:
    if r["name"] in labels:
        dx, dy = labels[r["name"]]
        name = "Homology only (best hit)" if r["name"].startswith("MMseqs2 + Foldseek") else r["name"]
        ax.annotate(name, (r["id50"], r["test_mean"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.2, color=INK, ha="right" if dx < 0 else "left")
x = np.array([r["id50"] for r in pts])
y = np.array([r["test_mean"] for r in pts])
rho = np.corrcoef(x.argsort().argsort(), y.argsort().argsort())[0, 1]
ax.text(0.02, 0.97, f"Spearman ρ = {rho:.2f} ({len(pts)} models)", transform=ax.transAxes, fontsize=7, va="top",
        color=QUIET)
ax.set_xlabel("id50 validation, per-enzyme F1")
ax.set_ylabel("Mean of six test sets, per-enzyme F1")
ax.grid(color=GRID, lw=0.7)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.5, loc="lower right", handletextpad=0.2)
fig.tight_layout()
fig.savefig(HERE / "fig6_id50_vs_test.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 4: difference from homology per test set
boot = json.loads((DATA / "bootstrap_vs_homology.json").read_text())["runs"]
show = [("esm2_3b_last", "ESM2-3B", "Sequence PLM"),
        ("esmc600_saemax", "ESMC-600M SAE, max-pooled", "PLM + SAE features"),
        ("prostt5_last", "ProstT5", "Structure-aware PLM"),
        ("saprotp70_last", "SaProt-650M, pLDDT-masked", "Structure-aware PLM"),
        ("ft_esmc600", "ESMC-600M, fine-tuned", "Fine-tuned PLM"),
        ("ens6_6b", "Six-head PLM ensemble", "PLM combination"),
        ("hyb_c3_msaf", "c3 + homology", "PLM + homology"),
        ("hyb_ens6_6b_msaf", "Six-head ensemble + homology", "PLM + homology")]
cols = TESTS + ["test_mean"]
fig, axes = plt.subplots(1, len(cols), figsize=(7.2, 2.9), sharey=True)
for ax, col in zip(axes, cols):
    for i, (run, name, cat) in enumerate(show):
        d, lo, hi = boot[run][col]
        sig = lo > 0 or hi < 0
        ax.plot([lo, hi], [i, i], color=CAT[cat], lw=1.4)
        ax.scatter([d], [i], s=20, color=CAT[cat] if sig else "white", edgecolor=CAT[cat], lw=1, zorder=3)
    ax.axvline(0, color=CAT["Homology only"], lw=0.9)
    ax.set_title(SET_LABEL[col], fontsize=7, fontweight="bold" if col == "test_mean" else "normal", loc="center")
    if col == "test_mean":
        ax.set_xlim(-0.06, 0.06)
    else:
        ax.set_xlim(-0.13, 0.13)
    ticks = [-0.04, 0, 0.04] if col == "test_mean" else [-0.1, 0, 0.1]
    ax.set_xticks(ticks, [f"{v:+.2f}".replace("0.", ".") if v else "0" for v in ticks])
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
axes[0].set_yticks(range(len(show)), [s[1] for s in show])
axes[0].set_ylim(len(show) - 0.5, -0.5)
axes[0].tick_params(axis="y", length=0)
fig.supxlabel("Difference in per-enzyme F1 from homology only (MMseqs2 + Foldseek-AFDB); 95% paired bootstrap interval",
              fontsize=7)
fig.tight_layout()
fig.savefig(HERE / "fig3_vs_homology.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 5: identity and rarity
ib = json.loads((DATA / "identity_bins.json").read_text())
rb = json.loads((DATA / "rarity_bins.json").read_text())
lines = [("hom_best_mmseqs_foldseek_afdb", "Homology only", CAT["Homology only"], "-", 1.8),
         ("esm2_3b_last", "ESM2-3B (best single PLM)", CAT["Sequence PLM"], "-", 1.3),
         ("prostt5_last", "ProstT5 (structure-aware PLM)", CAT["Structure-aware PLM"], "-", 1.3),
         ("ens6_6b", "Six-head PLM ensemble", CAT["PLM combination"], "-", 1.3),
         ("hyb_ens6_6b_msaf", "Six-head ensemble + homology", CAT["PLM + homology"], "--", 1.6)]
rar_keys = ["Homology only (MMseqs2 + Foldseek-AFDB)", "ESM2-3B (best single PLM)",
            "ProstT5 (best structure-aware PLM)", "Six-head PLM ensemble", "Six-head ensemble + homology"]
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), height_ratios=[2.3, 1], sharex="col")
bins = ib["tests"]["bins"]
xb = np.arange(len(bins))
for run, name, c, ls, lw in lines:
    ys = [np.nan if (b["bin"] == "no hit" and run.startswith("hom")) else b["f1"][run] for b in bins]
    axes[0, 0].plot(xb, ys, ls=ls, color=c, lw=lw, marker="o", ms=3.2, label=name)
axes[0, 0].text(0, 0.45, "no hit:\nsearch\ncannot call", ha="center", fontsize=6, color=QUIET)
axes[0, 0].set_ylabel("Per-enzyme F1")
axes[0, 0].legend(frameon=False, fontsize=6.3, loc="upper left")
axes[0, 0].set_title("a  By top-hit identity to training")
deficit = sum((1 - b["f1"]["hyb_ens6_6b_msaf"]) * b["n"] for b in bins)
share = [(1 - b["f1"]["hyb_ens6_6b_msaf"]) * b["n"] / deficit for b in bins]
axes[1, 0].bar(xb, share, width=0.55, color=CAT["PLM + homology"], alpha=0.35)
for i, s in enumerate(share):
    axes[1, 0].text(i, s + 0.02, f"{round(s * 100)}%", ha="center", fontsize=6.5)
axes[1, 0].set_xticks(xb, [f"{b['bin'].replace('>=', '≥')}\nn={b['n']:,}" for b in bins], fontsize=6.5)
axes[1, 0].set_xlabel("Top-hit identity (higher of MMseqs2 and Foldseek-AFDB)")
xr = np.arange(len(rb["bins"]))
for (run, name, c, ls, lw), key in zip(lines, rar_keys):
    axes[0, 1].plot(xr, rb["f1"][key], ls=ls, color=c, lw=lw, marker="o", ms=3.2)
axes[0, 1].set_title("b  By training examples of the rarest true EC")
axes[1, 1].bar(xr, rb["hybrid_error_share"], width=0.55, color=CAT["PLM + homology"], alpha=0.35)
for i, s in enumerate(rb["hybrid_error_share"]):
    axes[1, 1].text(i, s + 0.02, f"{round(s * 100)}%", ha="center", fontsize=6.5)
axes[1, 1].set_xticks(xr, [f"{b.replace('>=', '≥')}\nn={n:,}" for b, n in zip(rb["bins"], rb["n"])], fontsize=6.5)
axes[1, 1].set_xlabel("Training examples of the query's rarest EC")
for ax in axes[0]:
    ax.set_ylim(0.25, 0.97)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
for ax in axes[1]:
    ax.set_ylim(0, 0.68)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
axes[1, 0].set_ylabel("Share of hybrid's\nremaining error", fontsize=7)
fig.tight_layout()
fig.savefig(HERE / "fig4_where.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 6: metric and calls per enzyme
cal = {c["name"]: c for c in json.loads((DATA / "calibration.json").read_text())}
mean = lambda d: float(np.mean(list(d.values())))  # noqa: E731
grey, dark = CAT["Published method"], "#6b6b6b"
show6 = [("ESM2-BC", "ESM2-BC", grey), ("ESM2-BC, shift +5", "ESM2-BC, shift +5", grey),
         ("MIL-BC+ (retrained)", "MIL-BC+", dark), ("MIL-BC+ (retrained), shift +5", "MIL-BC+, shift +5", dark),
         ("MIL-BC+ (released)", "MIL-BC+ released", dark),
         ("MMseqs2 + Foldseek-AFDB, best hit", "Homology only", CAT["Homology only"]),
         ("Six-head ensemble + homology", "Best hybrid", CAT["PLM + homology"])]
POS = {  # label offsets (points) and alignment, per panel
    "sample": {"ESM2-BC": (-5, 3, "right"), "MIL-BC+": (0, -8, "center"), "MIL-BC+ released": (6, 0, "left"),
               "Homology only": (6, 0, "left"), "Best hybrid": (6, 0, "left"),
               "ESM2-BC, shift +5": (-6, 4, "right"), "MIL-BC+, shift +5": (-6, -5, "right")},
    "weighted": {"ESM2-BC": (0, -8, "center"), "MIL-BC+": (6, 0, "left"), "MIL-BC+ released": (6, 0, "left"),
                 "Homology only": (6, -3, "left"), "Best hybrid": (0, 7, "center"),
                 "ESM2-BC, shift +5": (-6, 5, "right"), "MIL-BC+, shift +5": (-6, -6, "right")}}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True, sharey=True)
for ax, key, title in [(axes[0], "sample", "a  Per-enzyme F1"), (axes[1], "weighted", "b  Support-weighted F1")]:
    for a_, b_ in [("ESM2-BC", "ESM2-BC, shift +5"), ("MIL-BC+ (retrained)", "MIL-BC+ (retrained), shift +5")]:
        ax.annotate("", xy=(mean(cal[b_]["calls"]), mean(cal[b_][key])),
                    xytext=(mean(cal[a_]["calls"]), mean(cal[a_][key])),
                    arrowprops=dict(arrowstyle="->", color="#c7c7c7", lw=0.8, shrinkA=5, shrinkB=5))
    for name, label, col in show6:
        x_, y_ = mean(cal[name]["calls"]), mean(cal[name][key])
        ax.scatter([x_], [y_], s=32, color=col, zorder=3)
        dx, dy, ha = POS[key][label]
        ax.annotate(f"{label} {y_:.3f}", (x_, y_), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.2, ha=ha, va="center", color=INK)
    ax.set_title(title)
    ax.set_xlabel("EC calls per enzyme (mean of six test sets)")
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
axes[0].set_ylabel("F1, mean of six test sets")
axes[0].set_xlim(0.8, 2.75)
axes[0].set_ylim(0.48, 0.665)
fig.tight_layout()
fig.savefig(HERE / "fig7_metrics.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: published methods vs homology
pb = json.loads((DATA / "published_bins.json").read_text())
pub_lines = [("Homology only", CAT["Homology only"], "-", 2.0, "o"),
             ("Best hybrid", CAT["PLM + homology"], "--", 1.8, "o"),
             ("ESM2-BC", "#6b6b6b", "-", 1.2, "s"),
             ("MIL-BC+", "#6b6b6b", "--", 1.2, "^"),
             ("MIL-CL+", "#ababab", "-", 1.0, "D"),
             ("EnzymeHunter", "#ababab", ":", 1.3, "v"),
             ("CLEAN", "#ababab", "--", 1.0, "<"),
             ("EnzHier", "#ababab", "-.", 1.0, ">"),
             ("CLEAN-Contact", "#ababab", (0, (1, 1)), 1.3, "P")]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.7), sharey=True)
for ax, key, title, xl in [(axes[0], "identity", "a  By top-hit identity to training", "Top-hit identity (MMseqs2 or Foldseek-AFDB)"),
                           (axes[1], "rarity", "b  By training examples of the rarest true EC", "Training examples of the query's rarest EC")]:
    bins_ = pb[key]
    xs = np.arange(len(bins_))
    for name, c, ls, lw, mk in pub_lines[::-1]:
        ys = [np.nan if (b["bin"] == "no hit" and name == "Homology only") else b["f1"][name] for b in bins_]
        ax.plot(xs, ys, ls=ls, color=c, lw=lw, marker=mk, ms=3.2, label=name,
                zorder=3 if name in ("Homology only", "Best hybrid") else 2)
    ax.set_xticks(xs, [f"{b['bin'].replace('>=', '≥')}\nn={b['n']:,}" for b in bins_], fontsize=6.5)
    ax.set_title(title)
    ax.set_xlabel(xl, fontsize=7)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Per-enzyme F1")
axes[0].set_ylim(0.15, 0.95)
axes[0].text(0, 0.18, "no search call", ha="center", fontsize=6, color=QUIET)
handles, labels_ = axes[1].get_legend_handles_labels()
order_ = [labels_.index(n) for n, *_ in pub_lines]
fig.legend([handles[i] for i in order_], [labels_[i] for i in order_], frameon=False, fontsize=6.5, ncol=5,
           loc="lower center", bbox_to_anchor=(0.5, 0.0), handlelength=2.6, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(HERE / "fig5_published.pdf")
plt.close(fig)
print(f"Spearman rho id50 vs test mean: {rho:.3f} over {len(pts)} models; wrote figures to {HERE}")
