# -*- coding: utf-8 -*-
"""
icml_plots.py
=============
Drop-in replacement for every plotting block in final_analysis_clean_v2.ipynb.

USAGE
-----
At the top of your notebook, after the analysis variables are computed, add:

    import importlib, icml_plots
    importlib.reload(icml_plots)
    icml_plots.generate_all(
        results=results,
        raw_runs=raw_runs,
        acc_overall=acc_overall,
        acc_by_cond=acc_by_cond,
        acc_hm=acc_hm,
        acc_mc=acc_mc,
        lang_pairs=lang_pairs,
        lang_summary=lang_summary,
        reason_pairs=reason_pairs,
        reason_summary=reason_summary,
        cond_summary=cond_summary,
        sig_all=sig_all,
        wr_slices=wr_slices,
        prov_long=prov_long,
        prov_table=prov_table,
        scale_overall=scale_overall,
        scale_country=scale_country,
        qwen_flip=qwen_flip,
        PLOTS=PLOTS,
        # constants from analysis_helpers:
        MODEL_ORDER=MODEL_ORDER,
        COUNTRY_ORDER=COUNTRY_ORDER,
        LANG_CONDITION_ORDER=LANG_CONDITION_ORDER,
        REASONING_PAIRS=REASONING_PAIRS,
        COUNTRY_LOCAL_LANG=COUNTRY_LOCAL_LANG,
    )

Each plot is also exposed as an individual function if you want to regenerate
one figure in isolation.

ICML 2026 layout targets
------------------------
* Full-width (two-column span): 6.75 in
* Single-column:                3.30 in
* Body font size:               10 pt  →  axis labels 10 pt, ticks 9 pt
* Minimum readable label:        8 pt
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# ──────────────────────────────────────────────────────────────────────────────
# 1. Global style
# ──────────────────────────────────────────────────────────────────────────────

# Wong (2011) colorblind-safe palette — 8 distinct hues
WONG = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#CC79A7",  # purple-pink
    "#F0E442",  # yellow  (use on dark backgrounds only)
    "#000000",  # black
]

# Country palette — three visually separated hues
CPAL = {"India": "#0072B2", "Turkey": "#D55E00", "Vietnam": "#009E73"}

# Full-width and single-column sizes
FW  = (6.75, 4.00)   # full-width figure (wide heatmaps / multi-panel)
FW_TALL = (6.75, 5.00)
SC  = (3.30, 3.00)   # single-column
SC_TALL = (3.30, 4.20)


def apply_icml_style() -> None:
    """
    Call once at notebook startup. Sets a clean, ICML-compatible rcParams.

    Does NOT require a LaTeX installation. If you have one, uncomment the
    text.usetex lines for exact paper-font matching.
    """
    mpl.rcParams.update({
        # --- fonts -----------------------------------------------------------
        # "text.usetex":          True,          # ← enable if LaTeX installed
        # "font.family":          "serif",
        # "font.serif":           ["Computer Modern Roman"],
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset":     "cm",            # CM math even without full LaTeX

        # --- sizes (match ICML 10 pt body) -----------------------------------
        "font.size":            12,
        "axes.titlesize":       14,
        "axes.titleweight":     "bold",
        "axes.labelsize":       12,
        "xtick.labelsize":       11,
        "ytick.labelsize":       11,
        "legend.fontsize":       11,
        "legend.title_fontsize": 11,

        # --- line / marker quality -------------------------------------------
        "lines.linewidth":       1.6,
        "lines.markersize":      6,
        "patch.linewidth":       0.6,

        # --- spines & grid ---------------------------------------------------
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":        0.8,
        "axes.grid":            True,
        "grid.color":           "#DCDCDC",
        "grid.linewidth":        0.6,
        "grid.alpha":            1.0,

        # --- color -----------------------------------------------------------
        "axes.prop_cycle":      mpl.cycler(color=WONG),

        # --- output ----------------------------------------------------------
        "figure.dpi":           200,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":    0.02,
        "figure.constrained_layout.use": True,
    })
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    sns.set_palette(WONG)


def _save(fig: plt.Figure, path_stem: Path, tight: bool = True) -> None:
    """Save as both PDF (vector) and PNG (raster) at 300 dpi."""
    kw = dict(bbox_inches="tight") if tight else {}
    fig.savefig(str(path_stem) + ".pdf", **kw)
    fig.savefig(str(path_stem) + ".png", dpi=300, **kw)
    plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Individual plot functions
# ──────────────────────────────────────────────────────────────────────────────

def plot_accuracy_heatmap(acc_hm: pd.DataFrame, PLOTS: Path) -> None:
    """
    Fig 1 — Accuracy (%) heat-map: Country × Model.
    Full-width, annotated, RdYlGn colormap centred at 50 %.
    """
    n_models = acc_hm.shape[1]
    fig, ax = plt.subplots(figsize=(min(6.75, 1.0 + 0.72 * n_models), 2.8))

    # Shorten model names if needed so they fit in columns
    short_cols = [c.replace("Qwen3.5-", "Q3.5-").replace("Gemma-4-", "G4-")
                  for c in acc_hm.columns]
    hm = acc_hm.copy()
    hm.columns = short_cols

    g = sns.heatmap(
        hm * 100,
        annot=True, annot_kws={"size": 10, "weight": "bold"},
        fmt=".1f",
        cmap="RdYlGn", center=50, vmin=30, vmax=75,
        linewidths=0.4, linecolor="#E8E8E8",
        cbar_kws={"label": "Accuracy (%)", "shrink": 0.85,
                  "ticks": [30, 40, 50, 60, 70]},
        ax=ax,
    )
    ax.set_title("Accuracy (%) by Country × Model", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35, labelsize=10)
    ax.tick_params(axis="y", rotation=0,  labelsize=11)
    g.collections[0].colorbar.ax.tick_params(labelsize=10)
    g.collections[0].colorbar.ax.set_ylabel("Accuracy (%)", fontsize=11)

    _save(fig, PLOTS / "accuracy_heatmap")


def plot_reasoning_slope(
    reason_pairs: dict,
    PLOTS: Path,
    COUNTRY_ORDER: list,
    COUNTRY_LOCAL_LANG: dict,
    resolve_language=None,          # callable from analysis_helpers
) -> None:
    if resolve_language is None:
        try:
            from analysis_helpers import resolve_language as helper_resolve_language
            resolve_language = helper_resolve_language
        except ImportError:
            def helper_resolve_language(country: str, lang_condition: str) -> str:
                return lang_condition
            resolve_language = helper_resolve_language
    """
    Fig 2 — Slope (bump) chart: Non-Reasoning vs Reasoning, per family.
    One panel per model family. Full-width, shared x-axis scale.
    """
    n = len(reason_pairs)
    if n == 0:
        return

    fig, axes = plt.subplots(
        1, n,
        figsize=(FW[0], 3.6 + 0.3 * max(3, sum(
            len(COUNTRY_ORDER) for _ in reason_pairs))),
        sharey=False,
    )
    if n == 1:
        axes = [axes]

    for idx, (label, pdf) in enumerate(reason_pairs.items()):
        ax = axes[idx]
        clean = pdf.query(
            "pair_available and context_type=='COUNTRY' and conflict_type=='NONE'"
        )
        if clean.empty:
            ax.set_visible(False)
            continue

        slope = (
            clean.groupby(["country", "lang_condition"])
            .agg(nr=("left_accuracy", "mean"), r=("right_accuracy", "mean"))
            .reset_index()
            .sort_values(["country", "lang_condition"])
        )

        yp = np.arange(len(slope))
        for y, row in zip(yp, slope.itertuples()):
            color = CPAL.get(row.country, "#555555")
            ax.plot([row.nr * 100, row.r * 100], [y, y],
                    color=color, lw=1.8, zorder=3, alpha=0.7)
            ax.scatter(row.nr * 100, y, color=color, s=52,
                       marker="o", zorder=5, edgecolors="white", linewidths=0.8)
            ax.scatter(row.r  * 100, y, color=color, s=52,
                       marker="D", zorder=5, edgecolors="white", linewidths=0.8)

        tick_labels = [
            f"{row.country}\n{resolve_language(row.country, row.lang_condition)}"
            for row in slope.itertuples()
        ]
        ax.set_yticks(yp)
        ax.set_yticklabels(tick_labels, fontsize=10)
        ax.set_xlabel("Accuracy (%)", fontsize=12)
        ax.set_title(label, fontsize=12, pad=6)
        ax.axvline(50, ls="--", lw=0.8, color="#888888", alpha=0.6)
        ax.set_xlim(25, 80)
        ax.grid(axis="x", alpha=0.5)
        ax.grid(axis="y", alpha=0)

        if idx == n - 1:
            legend_handles = [
                mpatches.Patch(color="#888888", label="circle = Non-Reasoning"),
                mpatches.Patch(color="#888888", label="diamond = Reasoning"),
            ] + [
                mpatches.Patch(color=CPAL[c], label=c)
                for c in COUNTRY_ORDER if c in CPAL
            ]
            ax.legend(
                handles=legend_handles,
                loc="lower right", fontsize=10,
                framealpha=0.9, edgecolor="#CCCCCC",
            )

    _save(fig, PLOTS / "reasoning_slope_all")


def plot_context_conflict_delta(cond_summary: pd.DataFrame, PLOTS: Path) -> None:
    """
    Fig 3 — Horizontal lollipop: accuracy Δ for context/conflict perturbations.
    Single-column-friendly.
    """
    df = cond_summary.sort_values("acc_delta").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(FW[0], 0.55 * len(df) + 1.4))

    colors = [WONG[0] if v >= 0 else WONG[3] for v in df["acc_delta"]]
    y = np.arange(len(df))

    # Stems
    ax.hlines(y, 0, df["acc_delta"] * 100,
              colors=colors, lw=2.0, alpha=0.85, zorder=2)
    # Dots
    ax.scatter(df["acc_delta"] * 100, y,
               color=colors, s=55, zorder=4,
               edgecolors="white", linewidths=0.8)

    # Value labels
    for i, (val, n) in enumerate(zip(df["acc_delta"] * 100, df["n_pairs"].astype(int))):
        offset = 0.5 if val >= 0 else -0.5
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, i, f"{val:+.1f} pp", va="center",
                ha=ha, fontsize=10, color="#333333")

    ax.axvline(0, color="#444444", lw=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["comparison"], fontsize=11)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:+.0f}"))
    ax.set_xlabel("Accuracy change (pp)", fontsize=12)
    ax.set_title("Context & Conflict Perturbation Effects", pad=8)
    ax.grid(axis="x", alpha=0.5)
    ax.grid(axis="y", alpha=0)

    # Colour legend
    ax.legend(
        handles=[
            mpatches.Patch(color=WONG[0], label="Beneficial (+)"),
            mpatches.Patch(color=WONG[3], label="Harmful (−)"),
        ],
        fontsize=10, loc="lower right", framealpha=0.9, edgecolor="#CCCCCC",
    )

    _save(fig, PLOTS / "context_conflict_delta")


def plot_provider_country(
    prov_long: pd.DataFrame,
    PLOTS: Path,
    COUNTRY_ORDER: list,
) -> None:
    """
    Fig 4 — Grouped bar: mean accuracy by provider × country.
    Full-width.
    """
    fig, ax = plt.subplots(figsize=FW)

    providers = prov_long["provider"].cat.categories.tolist() \
        if hasattr(prov_long["provider"], "cat") \
        else sorted(prov_long["provider"].unique())

    x = np.arange(len(providers))
    n_countries = len(COUNTRY_ORDER)
    width = 0.72 / n_countries
    offsets = np.linspace(-(n_countries - 1) / 2,
                          (n_countries - 1) / 2, n_countries) * width

    for i, country in enumerate(COUNTRY_ORDER):
        vals = [
            prov_long.loc[
                (prov_long["provider"] == p) & (prov_long["country"] == country),
                "acc"
            ].values
            for p in providers
        ]
        heights = [v[0] if len(v) else np.nan for v in vals]
        bars = ax.bar(x + offsets[i], heights, width=width * 0.92,
                      color=CPAL.get(country, WONG[i]),
                      edgecolor="white", linewidth=0.6,
                      label=country, zorder=3)
        for bar, h in zip(bars, heights):
            if np.isfinite(h) and h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6,
                        f"{h:.0f}", ha="center", va="bottom",
                        fontsize=9.5, color="#222222")

    ax.axhline(50, color="#888888", lw=0.9, ls="--", alpha=0.7, zorder=1,
               label="Chance (50 %)")
    ax.set_xticks(x)
    ax.set_xticklabels(providers, rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 78)
    ax.set_title("Cultural-Norm Accuracy by Provider × Country", pad=8)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
              fontsize=11, ncol=n_countries + 1, loc="upper right")
    ax.grid(axis="y", alpha=0.5)
    ax.grid(axis="x", alpha=0)

    _save(fig, PLOTS / "provider_country")


def plot_qwen_scaling(
    scale_overall: pd.DataFrame,
    scale_country: pd.DataFrame,
    PLOTS: Path,
    COUNTRY_ORDER: list,
) -> None:
    """
    Fig 5 — Log-scale line plot: Qwen3.5 size scaling (0.8 B → 2 B → 9 B).
    Single-column.
    """
    fig, ax = plt.subplots(figsize=SC_TALL)

    markers = ["o", "s", "^"]
    for i, country in enumerate(COUNTRY_ORDER):
        s = scale_country[scale_country["country"] == country].sort_values("params_B")
        ax.plot(s["params_B"], s["accuracy"] * 100,
                marker=markers[i], lw=1.8, ms=7,
                color=CPAL.get(country, WONG[i]),
                label=country, zorder=4)

    ov = scale_overall.sort_values("params_B")
    ax.plot(ov["params_B"], ov["accuracy"] * 100,
            marker="D", lw=2.2, ms=7,
            color="#444444", ls="--",
            label="Overall", zorder=5)

    ax.set_xscale("log")
    ax.set_xticks([0.8, 2, 9])
    ax.get_xaxis().set_major_formatter(mtick.FormatStrFormatter("%g B"))
    ax.set_xlabel("Parameters (log scale)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Qwen3.5 Size Scaling\non Cultural-Norm Task", pad=8)
    ax.set_ylim(30, 72)
    ax.axhline(50, color="#888888", lw=0.8, ls=":", alpha=0.7)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
              fontsize=11, loc="upper left")
    ax.grid(axis="both", alpha=0.4)

    _save(fig, PLOTS / "qwen_scaling")


def plot_forest_significance(sig_all: pd.DataFrame, PLOTS: Path) -> None:
    """
    Fig 6 — Forest plot: effect sizes with Holm significance stars.
    Full-width, colour-coded by family.
    """
    fp = sig_all.sort_values(["family", "acc_delta"]).reset_index(drop=True).copy()
    fp["label"] = fp["comparison"]

    families = fp["family"].unique().tolist()
    fam_colors = {f: WONG[i] for i, f in enumerate(families)}

    fig, ax = plt.subplots(figsize=(FW[0] + 0.5, 0.55 * len(fp) + 2.5))
    y = np.arange(len(fp))

    for i, row in fp.iterrows():
        col = fam_colors.get(row["family"], "#888888")
        val = row["acc_delta"] * 100
        ax.plot([0, val], [i, i], color=col, lw=2.0, alpha=0.8, zorder=2)
        ax.scatter(val, i, color=col, s=55, zorder=4,
                   edgecolors="white", linewidths=0.7)

        stars = (
            "***" if row["p_holm"] < 0.001
            else "**" if row["p_holm"] < 0.01
            else "*"  if row["p_holm"] < 0.05
            else "ns"
        )
        offset = 0.55 if val >= 0 else -0.55
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, i, stars,
                va="center", ha=ha,
                fontsize=11,
                color=col if stars != "ns" else "#AAAAAA")

    ax.set_yticks(y)
    ax.set_yticklabels(fp["label"], fontsize=12)
    ax.axvline(0, color="#444444", lw=1.0, zorder=1)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:+.0f}"))
    ax.set_xlabel("Accuracy Δ (pp; right − left)", fontsize=13)
    ax.set_title(
        "Effect Sizes Across Pre-Registered Comparisons\n"
        r"($^{*}$p<0.05, $^{**}$p<0.01, $^{***}$p<0.001 · Holm correction)",
        pad=10,
    )

    # Family legend
    legend_handles = [
        mpatches.Patch(color=fam_colors[f], label=f) for f in families
    ]
    ax.legend(handles=legend_handles, fontsize=12, loc="lower right",
              framealpha=0.9, edgecolor="#CCCCCC")

    ax.grid(axis="x", alpha=0.4)
    ax.grid(axis="y", alpha=0)

    _save(fig, PLOTS / "forest_significance")


def plot_conflict_vulnerability(
    viewB: pd.DataFrame,   # results filtered to lang_condition=='EN', context_type=='RULES'
    PLOTS: Path,
) -> None:
    """
    Fig 7 — Heatmap: provider × conflict-type accuracy delta under RULES context.
    """
    hm = (viewB.groupby(["provider", "conflict_type"])["accuracy"].mean()
          .unstack("conflict_type") * 100)
    if "NONE" not in hm.columns:
        return
    hm_delta = hm.subtract(hm["NONE"], axis=0).drop(columns=["NONE"])

    n_prov = len(hm_delta)
    fig, ax = plt.subplots(figsize=(max(4.5, 1.4 * len(hm_delta.columns) + 1.5),
                                    0.7 * n_prov + 1.8))
    g = sns.heatmap(
        hm_delta,
        annot=True, annot_kws={"size": 11, "weight": "bold"},
        fmt=".1f",
        cmap="RdBu_r", center=0,
        linewidths=0.4, linecolor="#E8E8E8",
        cbar_kws={"label": "Δ accuracy vs NONE (pp)", "shrink": 0.80},
        ax=ax,
    )
    ax.set_title("Conflict-Type Vulnerability\n(RULES context, English)", pad=8)
    ax.set_xlabel("Conflict type", fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=20, labelsize=11)
    ax.tick_params(axis="y", rotation=0,  labelsize=11)
    g.collections[0].colorbar.ax.tick_params(labelsize=10)
    g.collections[0].colorbar.ax.set_ylabel("Δ accuracy (pp)", fontsize=11)

    _save(fig, PLOTS / "conflict_vulnerability")


def plot_wrongrules_provider_country(
    wr_slices: pd.DataFrame,
    PLOTS: Path,
    COUNTRY_ORDER: list,
) -> None:
    """
    Fig 8 — Heatmap: provider × country accuracy drop under WRONG_RULES.
    """
    piv = (wr_slices.pivot(index="provider", columns="country", values="acc_delta")
           .reindex(columns=COUNTRY_ORDER) * 100)

    n_prov = len(piv)
    fig, ax = plt.subplots(figsize=(3.5 + 0.85 * len(COUNTRY_ORDER),
                                    0.72 * n_prov + 1.8))
    g = sns.heatmap(
        piv,
        annot=True, annot_kws={"size": 11, "weight": "bold"},
        fmt=".1f",
        cmap="RdBu_r", center=0,
        linewidths=0.4, linecolor="#E8E8E8",
        cbar_kws={"label": "Δ accuracy (pp)", "shrink": 0.80},
        ax=ax,
    )
    ax.set_title("Vulnerability to WRONG_RULES\nby Provider × Country (English, paired)", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0,  labelsize=11)
    ax.tick_params(axis="y", rotation=0,  labelsize=11)
    g.collections[0].colorbar.ax.tick_params(labelsize=10)
    g.collections[0].colorbar.ax.set_ylabel("Δ accuracy (pp)", fontsize=11)

    _save(fig, PLOTS / "wrongrules_provider_country")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Convenience: generate all figures in one call
# ──────────────────────────────────────────────────────────────────────────────

def generate_all(
    *,
    results,
    raw_runs,
    acc_overall,
    acc_by_cond,
    acc_hm,
    acc_mc=None,
    lang_pairs,
    lang_summary,
    reason_pairs,
    reason_summary,
    cond_summary,
    sig_all,
    wr_slices,
    prov_long,
    prov_table,
    scale_overall,
    scale_country,
    qwen_flip,
    PLOTS: Path,
    MODEL_ORDER,
    COUNTRY_ORDER,
    LANG_CONDITION_ORDER,
    REASONING_PAIRS,
    COUNTRY_LOCAL_LANG,
    resolve_language=None,         # pass from analysis_helpers
) -> None:
    if resolve_language is None:
        try:
            from analysis_helpers import resolve_language as helper_resolve_language
            resolve_language = helper_resolve_language
        except ImportError:
            def helper_resolve_language(country: str, lang_condition: str) -> str:
                return lang_condition
            resolve_language = helper_resolve_language
    """Generate and save all ICML-ready figures."""

    apply_icml_style()

    print("Generating publication-quality figures for ICML 2026 …\n")

    print("  [1/8] Accuracy heatmap …")
    plot_accuracy_heatmap(acc_hm, PLOTS)

    print("  [2/8] Reasoning slope chart …")
    plot_reasoning_slope(reason_pairs, PLOTS,
                         COUNTRY_ORDER, COUNTRY_LOCAL_LANG, resolve_language)

    print("  [3/8] Context/conflict delta lollipop …")
    plot_context_conflict_delta(cond_summary, PLOTS)

    print("  [4/8] Provider × country grouped bar …")
    plot_provider_country(prov_long, PLOTS, COUNTRY_ORDER)

    print("  [5/8] Qwen3.5 scaling …")
    plot_qwen_scaling(scale_overall, scale_country, PLOTS, COUNTRY_ORDER)

    print("  [6/8] Forest / significance plot …")
    plot_forest_significance(sig_all, PLOTS)

    print("  [7/8] Conflict vulnerability heatmap …")
    conflict_focus = results[results["lang_condition"] == "EN"].copy()
    viewB = conflict_focus[conflict_focus["context_type"] == "RULES"]
    plot_conflict_vulnerability(viewB, PLOTS)

    print("  [8/8] WRONG_RULES provider × country heatmap …")
    plot_wrongrules_provider_country(wr_slices, PLOTS, COUNTRY_ORDER)

    print(f"\nDone — PDF + PNG written to {PLOTS}/")
    print("All figures target ICML 2026 two-column layout (6.75″ full-width).")