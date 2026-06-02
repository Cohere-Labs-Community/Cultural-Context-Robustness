#!/usr/bin/env python3
"""
Cultural Norm Robustness Analysis script.
Regenerates all analysis numbers, LaTeX tables, and paper macros
needed for paper.tex using the aggregated model results.
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure final_analysis is in import path
sys.path.append(str(Path(__file__).parent))

import analysis_helpers
from analysis_helpers import (
    load_runs,
    aggregate,
    build_pairs,
    summarize_pairs,
    resolve_language,
    macro_f1,
    MODEL_ORDER,
    COUNTRY_ORDER,
    LANG_CONDITION_ORDER,
    ACTUAL_LANGUAGE_MAP,
    COUNTRY_LOCAL_LANG,
    REASONING_PAIRS,
    COND_PAIR_IDX,
    LANGCOND_PAIR_IDX,
    MODEL_PAIR_IDX,
    PAIR_VCOLS,
    PROVIDER,
)

# Output paths
WORKDIR = Path(__file__).parent
OUT = WORKDIR / "final_analysis_outputs"
PAPER = OUT / "paper"
OUT.mkdir(parents=True, exist_ok=True)
PAPER.mkdir(parents=True, exist_ok=True)

DISPLAY_MAP = {
    'basic_etiquette': 'Etiquette',
    'visiting': 'Visiting',
    'eating': 'Eating',
    'gift_giving': 'Gift-giving',
    'manners_in_vietnam': 'Manners in Vietnam',
    'gifts': 'Gifts'
}

def main():
    print("=" * 80)
    # 1. LOAD DATA
    print("1. Loading aggregated runs from CSV files...")
    paths = sorted((WORKDIR / "final_results").glob("aggregated_all_models_*.csv"))
    if len(paths) != 3:
        print(f"Error: Expected 3 CSV files, found {len(paths)} at final_results/")
        sys.exit(1)
        
    raw_runs_all = load_runs(paths, WORKDIR)
    
    # 2. SEPARATE UNPARSEABLES
    unparseable_mask = raw_runs_all["prediction"].isna()
    raw_runs_unp = raw_runs_all[unparseable_mask].copy()
    raw_runs = raw_runs_all[~unparseable_mask].copy()
    
    n_total = len(raw_runs_all)
    n_val = len(raw_runs)
    n_unp = len(raw_runs_unp)
    
    print(f"   - Total raw runs: {n_total:,}")
    print(f"   - Valid parsed  : {n_val:,} ({n_val/n_total*100:.2f}%)")
    print(f"   - Unparseable   : {n_unp:,} ({n_unp/n_total*100:.2f}%)")
    
    # Calculate parse-rate breakdown by country x model
    parse_breakdown = (
        raw_runs_all
        .assign(is_unp=unparseable_mask.values)
        .groupby(["country", "model"])
        .agg(
            total=("scenario_id", "size"),
            unp=("is_unp", "sum"),
        )
        .assign(
            valid=lambda d: d["total"] - d["unp"],
            parse_rate=lambda d: (d["valid"] / d["total"]).round(4),
        )
        .reset_index()
    )
    has_unp = parse_breakdown[parse_breakdown["unp"] > 0].copy()
    parse_breakdown.to_csv(OUT / "parse_rate_by_model_country.csv", index=False)
    
    # 3. RUN AGGREGATION (Majority voting over 5 runs per condition)
    print("\n2. Aggregating runs to conditions...")
    results = aggregate(raw_runs)
    print(f"   - Final aggregated conditions: {len(results):,}")
    
    # 4. OVERALL MODEL RANKING
    acc_overall = results.groupby("model")["accuracy"].mean().reindex(MODEL_ORDER).dropna().sort_values(ascending=False)
    acc_mc = (
        results.groupby(["model", "country"])["accuracy"].mean()
        .unstack("country")
        .reindex(index=MODEL_ORDER)
        .reindex(columns=COUNTRY_ORDER)
        .dropna(how="all")
    )
    
    # 4b. PROVIDER-LEVEL MACRO F1 (for paper Table 1: tab:provider)
    print("\n2b. Computing provider-level Macro F1...")
    valid = results[results["prediction"].notna() & results["gold_label"].notna()].copy()
    provider_order = ["Google Gemma", "Cohere", "Qwen"]
    provider_label_map = {"Gemma": "Google Gemma", "Cohere": "Cohere", "Qwen": "Qwen"}
    provider_rows = []
    for prov_key in ["Gemma", "Cohere", "Qwen"]:
        d = valid[valid["provider"] == prov_key]
        mf1_overall = macro_f1(d["gold_label"], d["prediction"])
        row = {"provider": provider_label_map[prov_key], "macro_f1": mf1_overall}
        for country in COUNTRY_ORDER:
            dc = d[d["country"] == country]
            row[f"acc_{country}"] = dc["accuracy"].mean() if len(dc) > 0 else np.nan
        provider_rows.append(row)
        print(f"   - {provider_label_map[prov_key]}: Macro F1 = {mf1_overall*100:.1f}%")
    df_provider = pd.DataFrame(provider_rows)
    
    # 5. LANGUAGE ROBUSTNESS DELTAS
    print("\n3. Running language robustness comparisons...")
    lang_comps = [
        ("EN", "EN_PARA", "EN → EN paraphrase"),
        ("EN", "LOCAL", "EN → Local language"),
        ("LOCAL", "LOCAL_PARA", "Local → Local paraphrase"),
        ("EN", "LOCAL_PARA", "EN → Local paraphrase"),
        ("EN", "MISMATCH", "EN → Mismatched language"),
        ("LOCAL", "MISMATCH", "Local → Mismatched language"),
    ]
    lang_filters = {"context_type": "COUNTRY", "conflict_type": "NONE"}
    lang_pairs = {}
    for l, r, desc in lang_comps:
        if {l, r}.issubset(results["lang_condition"].unique()):
            lang_pairs[desc] = build_pairs(results, LANGCOND_PAIR_IDX, "lang_condition", l, r, PAIR_VCOLS, lang_filters)
            
    lang_summary = pd.concat([
        summarize_pairs(p).assign(comparison=k) for k, p in lang_pairs.items()
    ], ignore_index=True).sort_values("flip_rate", ascending=False)
    
    # 6. CONTEXT / CONFLICT ROBUSTNESS DELTAS
    print("4. Running context and conflict robustness comparisons...")
    cond_comps = [
        ("COUNTRY_NONE", "NONE_NONE", "Country → No context"),
        ("COUNTRY_NONE", "VALUES_NONE", "Country → Values"),
        ("COUNTRY_NONE", "RULES_NONE", "Country → Rules"),
        ("RULES_NONE", "RULES_WRONG_RULES", "Rules → Wrong rules"),
        ("RULES_NONE", "RULES_WRONG_COUNTRY", "Rules → Wrong country"),
        ("COUNTRY_NONE", "COUNTRY_MADE_UP", "Country → Made-up country"),
    ]
    cond_pairs = {}
    for l, r, desc in cond_comps:
        if {l, r}.issubset(results["condition_label"].unique()):
            cond_pairs[desc] = build_pairs(results, COND_PAIR_IDX, "condition_label", l, r, PAIR_VCOLS, {"lang_condition": "EN"})
            
    cond_summary = pd.concat([
        summarize_pairs(p).assign(comparison=k) for k, p in cond_pairs.items()
    ], ignore_index=True).sort_values("acc_delta")
    
    # 7. REASONING TOGGLE WITHIN-FAMILY COMPARISONS
    print("5. Running reasoning family comparisons...")
    reason_pairs = {}
    for l, r in REASONING_PAIRS:
        if {l, r}.issubset(results["model"].unique()):
            reason_pairs[f"{l} → {r}"] = build_pairs(results, MODEL_PAIR_IDX, "model", l, r, PAIR_VCOLS)
            
    reason_summary = pd.concat([
        summarize_pairs(p).assign(comparison=k) for k, p in reason_pairs.items()
    ], ignore_index=True)
    
    # 8. TINY-AYA DOMAIN-LEVEL ANALYSIS
    print("6. Mapping domains and computing Tiny-Aya domain-level stats...")
    domain_csv_path = WORKDIR.parent / "notebooks" / "merged_experiment_results.csv"
    if domain_csv_path.exists():
        df_merged = pd.read_csv(domain_csv_path)
        id_to_domain = df_merged.groupby('scenario_id (Unique ID from NormAd dataset)')['domain (etiquette, eating, visiting etc)'].first().to_dict()
        results['domain'] = results['scenario_id'].astype(str).map({str(k): v for k, v in id_to_domain.items()})
        
        ta_models = ['Global', 'Fire', 'Earth', 'Water']
        df_ta = results[results['model'].isin(ta_models)].copy()
        
        df_ta['domain_paper'] = df_ta['domain']
        
        PAPER_DOMAINS = [
            'visiting',
            'gift_giving',
            'basic_etiquette',
            'eating',
            'manners_in_vietnam',
            'gifts'
        ]
        
        domain_table = []
        for d in PAPER_DOMAINS:
            row = {'Domain': d}
            # Global baseline
            row['Global'] = df_ta[(df_ta['domain_paper'] == d) & (df_ta['model'] == 'Global')]['accuracy'].mean()
            # Turkey (Earth model in Turkey)
            row['Turkey'] = df_ta[(df_ta['domain_paper'] == d) & (df_ta['model'] == 'Earth') & (df_ta['country'] == 'Turkey')]['accuracy'].mean()
            # Vietnam (Water model in Vietnam)
            row['Vietnam'] = df_ta[(df_ta['domain_paper'] == d) & (df_ta['model'] == 'Water') & (df_ta['country'] == 'Vietnam')]['accuracy'].mean()
            # India (Fire model in India)
            row['India'] = df_ta[(df_ta['domain_paper'] == d) & (df_ta['model'] == 'Fire') & (df_ta['country'] == 'India')]['accuracy'].mean()
            
            # Regional Average (ignoring NaN)
            vals = [row[c] for c in ['Turkey', 'Vietnam', 'India'] if pd.notna(row[c])]
            row['Regional'] = np.mean(vals) if vals else np.nan
            
            # Compute Delta using difference of rounded values to match paper.tex Table 5 perfectly
            if pd.notna(row['Regional']) and pd.notna(row['Global']):
                reg_pct = round(row['Regional'] * 100, 1)
                glob_pct = round(row['Global'] * 100, 1)
                row['Delta'] = (reg_pct - glob_pct) / 100.0
            else:
                row['Delta'] = np.nan
            
            domain_table.append(row)
            
        df_domain = pd.DataFrame(domain_table)
    else:
        print("Warning: Domain mapping file not found at notebooks/merged_experiment_results.csv! Skipping domain stats.")
        df_domain = None
        
    # 9. GENERATE LATEX MACROS AND TABLES
    print("\n7. Writing LaTeX macros and tables to final_analysis_outputs/paper/...")
    
    # 9.1 Macros
    macro = [
        "% Auto-generated — do not edit by hand.",
        f"\\newcommand{{\\TotalRawRuns}}{{{n_val:,}}}",
        f"\\newcommand{{\\TotalConditions}}{{{len(results):,}}}",
        f"\\newcommand{{\\NumModels}}{{{results['model'].nunique()}}}",
    ]
    for m, a in acc_overall.items():
        safe = str(m).replace("-", "").replace(".", "").replace(" ", "")
        macro.append(f"\\newcommand{{\\Acc{safe}}}{{{a*100:.1f}\\%}}")
    for country in COUNTRY_ORDER:
        ca = results[results["country"] == country].groupby("model")["accuracy"].mean()
        best_m, best_a = ca.idxmax(), ca.max()
        sc = country.replace(" ", "")
        macro.append(f"\\newcommand{{\\BestModel{sc}}}{{{best_m}}}")
        macro.append(f"\\newcommand{{\\BestAcc{sc}}}{{{best_a*100:.1f}\\%}}")
    for _, r in cond_summary.iterrows():
        tag = r.comparison.replace(" ", "").replace("→", "To").replace("_", "")
        macro.append(f"\\newcommand{{\\Cond{tag}Flip}}{{{r.flip_rate*100:.1f}\\%}}")
        macro.append(f"\\newcommand{{\\Cond{tag}Delta}}{{{r.acc_delta*100:+.1f}}}")
        macro.append(f"\\newcommand{{\\Cond{tag}CTW}}{{{r.c2w_rate*100:.1f}\\%}}")
        macro.append(f"\\newcommand{{\\Cond{tag}WTC}}{{{r.w2c_rate*100:.1f}\\%}}")
    for _, r in reason_summary.iterrows():
        tag = r.comparison.replace(" ", "").replace("→", "To").replace("-", "").replace(".", "")
        macro.append(f"\\newcommand{{\\Reason{tag}Flip}}{{{r.flip_rate*100:.1f}\\%}}")
        macro.append(f"\\newcommand{{\\Reason{tag}Delta}}{{{r.acc_delta*100:+.1f}}}")
        macro.append(f"\\newcommand{{\\Reason{tag}NRAcc}}{{{r.left_acc*100:.1f}\\%}}")
        macro.append(f"\\newcommand{{\\Reason{tag}RAcc}}{{{r.right_acc*100:.1f}\\%}}")
    (PAPER / "paper_macros.tex").write_text("\n".join(macro) + "\n")
    
    # 9.2a Table: Provider-level Macro F1 + per-country accuracy (paper Table 1)
    tbl = ["\\begin{table}[t]",
           "\\caption{Macro F1 (\\%) and per-country accuracy (\\%) by provider. "
           "Provider family is the clearest aggregate axis, but later sections show "
           "that clean accuracy and robustness to perturbations are distinct. "
           "The Qwen aggregate includes Qwen3.5 checkpoints at 0.8B, 2B, and 9B "
           "parameters; the Gemma aggregate includes Gemma-4-31B checkpoints.}",
           "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{5pt}",
           "\\begin{tabular}{lrrrr}", "\\toprule",
           "Provider & Macro F1 & " + " & ".join(COUNTRY_ORDER) + " \\\\", "\\midrule"]
    for _, r in df_provider.iterrows():
        cells = [r["provider"], f"{r['macro_f1']*100:.1f}"]
        for c in COUNTRY_ORDER:
            v = r[f"acc_{c}"]
            cells.append(f"{v*100:.1f}" if pd.notna(v) else "---")
        tbl.append(" & ".join(cells) + " \\\\")
    tbl += ["\\bottomrule", "\\end{tabular}",
            "\\label{tab:provider}", "\\end{table}"]
    (PAPER / "tab_provider.tex").write_text("\n".join(tbl) + "\n")
    
    # 9.2b Table: Model Ranking (full per-model accuracy table)
    tbl = ["\\begin{table*}[t]", "\\centering\\footnotesize", "\\setlength{\\tabcolsep}{5pt}",
           "\\begin{tabular}{l" + "r" * (1 + len(COUNTRY_ORDER)) + "}", "\\toprule",
           "Model & Overall & " + " & ".join(COUNTRY_ORDER) + " \\\\", "\\midrule"]
    for m in MODEL_ORDER:
        if m not in acc_overall.index: continue
        cells = [m, f"{acc_overall[m]*100:.1f}"]
        for c in COUNTRY_ORDER:
            v = acc_mc.loc[m, c] if m in acc_mc.index and pd.notna(acc_mc.loc[m, c]) else np.nan
            cells.append(f"{v*100:.1f}" if pd.notna(v) else "---")
        tbl.append(" & ".join(cells) + " \\\\")
    tbl += ["\\bottomrule", "\\end{tabular}",
            f"\\caption{{Accuracy (\\%) by model and country ({len(results):,} conditions from {n_val:,} raw runs).}}",
            "\\label{tab:accuracy-by-model}", "\\end{table*}"]
    (PAPER / "tab_accuracy_by_model.tex").write_text("\n".join(tbl) + "\n")
    
    # 9.3 Table 2: Context/Conflict
    tbl = ["\\begin{table*}[t]", "\\centering\\footnotesize", "\\setlength{\\tabcolsep}{4.5pt}",
           "\\begin{tabularx}{\\textwidth}{Xrrrrr}", "\\toprule",
           "Comparison & Pairs & Flip & Acc. $\\Delta$ & C2W & W2C \\\\", "\\midrule"]
    for _, r in cond_summary.iterrows():
        tbl.append(f"{r.comparison} & {int(r.n_pairs)} & {r.flip_rate*100:.1f}\\% & ${r.acc_delta*100:+.1f}$ pp & {r.c2w_rate*100:.1f}\\% & {r.w2c_rate*100:.1f}\\% \\\\")
    tbl += ["\\bottomrule", "\\end{tabularx}", "\\caption{Context and conflict effects.}", "\\label{tab:context-conflict}", "\\end{table*}"]
    (PAPER / "tab_context_conflict.tex").write_text("\n".join(tbl) + "\n")
    
    # 9.4 Table 3: Language perturbation
    tbl = ["\\begin{table*}[t]", "\\centering\\footnotesize", "\\setlength{\\tabcolsep}{5pt}",
           "\\begin{tabularx}{\\textwidth}{Xrrrrr}", "\\toprule",
           "Comparison & Pairs & Flip & Acc. $\\Delta$ & C2W & W2C \\\\", "\\midrule"]
    for _, r in lang_summary.iterrows():
        tbl.append(f"{r.comparison} & {int(r.n_pairs)} & {r.flip_rate*100:.1f}\\% & ${r.acc_delta*100:+.1f}$ pp & {r.c2w_rate*100:.1f}\\% & {r.w2c_rate*100:.1f}\\% \\\\")
    tbl += ["\\bottomrule", "\\end{tabularx}", "\\caption{Language perturbation effects. EN=English, LOCAL=Hindi/Turkish/Vietnamese, MISMATCH=unrelated language.}", "\\label{tab:language}", "\\end{table*}"]
    (PAPER / "tab_language.tex").write_text("\n".join(tbl) + "\n")
    
    # 9.5 Table 4: Reasoning toggle
    tbl = ["\\begin{table*}[t]", "\\centering\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
           "\\begin{tabularx}{\\textwidth}{Xrrrrrrr}", "\\toprule",
           "Comparison & Pairs & Flip & Acc. $\\Delta$ & C2W & W2C & NR Acc & R Acc \\\\", "\\midrule"]
    for _, r in reason_summary.iterrows():
        c = r.comparison.replace("_", "\\_")
        tbl.append(f"{c} & {int(r.n_pairs)} & {r.flip_rate*100:.1f}\\% & ${r.acc_delta*100:+.1f}$ pp & {r.c2w_rate*100:.1f}\\% & {r.w2c_rate*100:.1f}\\% & {r.left_acc*100:.1f}\\% & {r.right_acc*100:.1f}\\% \\\\")
    tbl += ["\\bottomrule", "\\end{tabularx}", "\\caption{Reasoning toggle: within-family comparisons for all three model families.}", "\\label{tab:reasoning}", "\\end{table*}"]
    (PAPER / "tab_reasoning.tex").write_text("\n".join(tbl) + "\n")
    
    # 9.6 Table 5: Tiny-Aya Domains
    if df_domain is not None:
        tbl = ["\\begin{table}[t]", "\\caption{Domain-level accuracy (\\%) for Tiny-Aya global and regional variants.}",
               "\\label{tab:tinyaya_domain}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{5pt}",
               "\\begin{tabular}{lrrr}", "\\toprule",
               "Domain & Global & Regional & Regional--Global \\\\", "\\midrule"]
        # Print only the 6 main non-NaN domains
        for _, r in df_domain.dropna(subset=['Regional', 'Global']).iterrows():
            dname = DISPLAY_MAP.get(r.Domain, r.Domain)
            delta_val = r.Delta * 100
            delta_str = "0.0" if abs(delta_val) < 1e-9 else f"{delta_val:+.1f}"
            tbl.append(f"{dname:20s} & {r.Global*100:.1f} & {r.Regional*100:.1f} & {delta_str} \\\\")
        tbl += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        (PAPER / "tab_tinyaya_domain.tex").write_text("\n".join(tbl) + "\n")
        
    # 9.7 Parse Rate Table (Appendix)
    if not has_unp.empty:
        tbl = [
            "\\begin{table}[t]", "\\centering\\footnotesize",
            "\\setlength{\\tabcolsep}{5pt}",
            "\\begin{tabular}{llrrrr}", "\\toprule",
            "Country & Model & Total & Unparseable & Valid & Parse Rate \\\\", "\\midrule",
        ]
        for _, r in has_unp.sort_values(["country", "model"]).iterrows():
            flag = "$^{*}$" if r["parse_rate"] < 0.90 else ""
            tbl.append(f"{r['country']} & {r['model']} & {int(r['total'])} & {int(r['unp'])} & "
                       f"{int(r['valid'])} & {r['parse_rate']*100:.1f}\\%{flag} \\\\")
        tbl += [
            "\\bottomrule",
            "\\multicolumn{6}{l}{\\footnotesize $^{*}$ Parse rate below 90\\%.}\\\\",
            "\\end{tabular}",
            f"\\caption{{Parse rates by model and country. {n_unp:,} runs excluded "
            f"({n_unp/n_total*100:.2f}\\% of {n_total:,} total).}}",
            "\\label{tab:parse-rate}", "\\end{table}",
        ]
        (PAPER / "tab_parse_rate.tex").write_text("\n".join(tbl) + "\n")
        
    # Excel Writer sheet
    with pd.ExcelWriter(OUT / "phase3_results_sheet.xlsx", engine="openpyxl") as w:
        acc_overall.reset_index().rename(columns={"index":"Model","accuracy":"Accuracy"}).to_excel(w, sheet_name="Overall Ranking", index=False)
        (acc_mc*100).round(1).to_excel(w, sheet_name="Accuracy by Country")
        lang_summary.to_excel(w, sheet_name="Lang Cond Robustness", index=False)
        reason_summary.to_excel(w, sheet_name="Reasoning Toggle", index=False)
        cond_summary.to_excel(w, sheet_name="Context Conflict", index=False)
        if df_domain is not None:
            df_domain.to_excel(w, sheet_name="TinyAya Domains", index=False)
            
    print("   - All outputs generated successfully in final_analysis_outputs/!")
    print("=" * 80)
    
    # 10. PRINT TEXT REPORT OF CRUCIAL NUMBERS
    print("\n" + "="*80)
    print("                     PAPER KEY NUMBERS REPORT")
    print("="*80)
    print(f"\n1. ABSTRACT & INTRODUCTION STATS:")
    print(f"   - Total valid raw runs   : {n_val:,}")
    print(f"   - Total conditions       : {len(results):,}")
    print(f"   - Model count            : {results['model'].nunique()}")
    
    print(f"\n1b. PROVIDER MACRO F1 (Table 1):")
    for _, r in df_provider.iterrows():
        accs = ", ".join(f"{c}={r[f'acc_{c}']*100:.1f}%" for c in COUNTRY_ORDER if pd.notna(r[f'acc_{c}']))
        print(f"   - {r['provider']:15s}: Macro F1={r['macro_f1']*100:.1f}%, {accs}")
    
    cs = cond_summary.set_index("comparison")
    if "Country → Rules" in cs.index:
        r = cs.loc["Country → Rules"]
        print(f"   - Rules accuracy increase: {r.acc_delta*100:+.1f} pp (flip={r.flip_rate*100:.1f}%)")
    if "Rules → Wrong rules" in cs.index:
        r = cs.loc["Rules → Wrong rules"]
        print(f"   - Wrong rules accuracy drop: {r.acc_delta*100:+.1f} pp")
        print(f"     Wrong rules C2W rate    : {r.c2w_rate*100:.1f}%")
        print(f"     Wrong rules W2C rate    : {r.w2c_rate*100:.1f}%")
        
    print(f"\n2. SECTION 4.3 (Reasoning Toggle Overall):")
    for _, r in reason_summary.iterrows():
        print(f"   - {r.comparison:40s}: flip={r.flip_rate*100:.1f}%, Δ={r.acc_delta*100:+.1f} pp (NR={r.left_acc*100:.1f}%, R={r.right_acc*100:.1f}%)")
        
    if df_domain is not None:
        print(f"\n3. SECTION 4.5 (Tiny-Aya Regional Specialization deltas):")
        for _, r in df_domain.dropna(subset=['Regional', 'Global']).iterrows():
            dname = DISPLAY_MAP.get(r.Domain, r.Domain)
            print(f"   - {dname:24s}: Global={r.Global*100:5.1f}%, Regional={r.Regional*100:5.1f}%, Delta={r.Delta*100:+5.1f} pp")
            
    print("\n4. PARSING ISSUES / ERROR RATES:")
    print(f"   - Excluded runs          : {n_unp:,} ({n_unp/n_total*100:.2f}%)")
    if not has_unp.empty:
        for _, r in has_unp.sort_values("parse_rate").iterrows():
            print(f"   - Model: {r['model']:26s} | Country: {r['country']:8s} | Parse Rate: {r['parse_rate']*100:.1f}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
