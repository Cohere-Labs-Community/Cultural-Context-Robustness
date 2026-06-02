"""Shared utilities for Phase-3 cultural-norm robustness analysis."""
from __future__ import annotations
import math, os, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

LABEL_ORDER = ["yes","no","neutral"]
LABEL_RANK = {l:i for i,l in enumerate(LABEL_ORDER)}
COUNTRY_ORDER = ["India","Turkey","Vietnam"]
# The CSV 'language' column stores perturbation CONDITIONS, not actual languages.
# EN/EN_PARA = English, LOCAL/LOCAL_PARA = country's local language, MISMATCH = unrelated language.
LANG_CONDITION_ORDER = ["EN","EN_PARA","LOCAL","LOCAL_PARA","MISMATCH"]
CONTEXT_ORDER = ["NONE","COUNTRY","VALUES","RULES"]
CONFLICT_ORDER = ["NONE","WRONG_RULES","WRONG_COUNTRY","MADE_UP"]

# Mapping: (country, condition) → actual language
ACTUAL_LANGUAGE_MAP = {
    ("India","EN"): "English", ("India","EN_PARA"): "English (para)",
    ("India","LOCAL"): "Hindi", ("India","LOCAL_PARA"): "Hindi (para)",
    ("India","MISMATCH"): "Mismatch",
    ("Turkey","EN"): "English", ("Turkey","EN_PARA"): "English (para)",
    ("Turkey","LOCAL"): "Turkish", ("Turkey","LOCAL_PARA"): "Turkish (para)",
    ("Turkey","MISMATCH"): "Mismatch",
    ("Vietnam","EN"): "English", ("Vietnam","EN_PARA"): "English (para)",
    ("Vietnam","LOCAL"): "Vietnamese", ("Vietnam","LOCAL_PARA"): "Vietnamese (para)",
    ("Vietnam","MISMATCH"): "Mismatch",
}
COUNTRY_LOCAL_LANG = {"India":"Hindi", "Turkey":"Turkish", "Vietnam":"Vietnamese"}

def resolve_language(country: str, lang_condition: str) -> str:
    """Map (country, language_condition) → actual language name."""
    return ACTUAL_LANGUAGE_MAP.get((country, lang_condition), lang_condition)
MODEL_ORDER = [
    "Global","Fire","Earth","Water",
    "Qwen3.5-0.8B","Qwen3.5-2B",
    "Qwen3.5-9B","Qwen3.5-9B-Reasoning",
    "Gemma-4-31B","Gemma-4-31B-Reasoning",
    "Command-a","Command-a-Reasoning",
]
REASONING_PAIRS = [
    ("Command-a","Command-a-Reasoning"),
    ("Gemma-4-31B","Gemma-4-31B-Reasoning"),
    ("Qwen3.5-9B","Qwen3.5-9B-Reasoning"),
]
CONDITION_KEY = ["country","model","model_id","scenario_id","lang_condition","context_type","conflict_type"]
LANGCOND_PAIR_IDX = ["country","scenario_id","model"]
MODEL_PAIR_IDX = ["country","scenario_id","lang_condition","context_type","conflict_type"]
COND_PAIR_IDX = ["country","scenario_id","model"]
PAIR_VCOLS = ["prediction","accuracy","confidence_pct","agreement_rate","parse_rate","reasoning_length_mean"]
EXPECTED_RUNS = 5

def safe_text(v):
    return "" if v is None or pd.isna(v) else str(v).strip()

def normalize_label(v):
    t = safe_text(v).lower()
    if not t: return np.nan
    if t in {"yes","1"} or t.startswith("yes"): return "yes"
    if t in {"no","2"} or t.startswith("no"): return "no"
    if t in {"neutral","neither","3"} or "neutral" in t or "neither" in t: return "neutral"
    return np.nan

_MMAP = {
    "global":"Global","tiny-aya-global":"Global",
    "fire":"Fire","tiny-aya-fire":"Fire",
    "earth":"Earth","tiny-aya-earth":"Earth",
    "water":"Water","tiny-aya-water":"Water",
    "command-a":"Command-a","command-a-03-2025":"Command-a",
    "command-a-reasoning":"Command-a-Reasoning","command-a-reasoning-08-2025":"Command-a-Reasoning",
    "gemma-4-31b-nr":"Gemma-4-31B","gemma-4-31b-r":"Gemma-4-31B-Reasoning",
    "qwen3.5-0.8b":"Qwen3.5-0.8B","qwen3.5-2b":"Qwen3.5-2B",
    "qwen3.5-9b-nr":"Qwen3.5-9B","qwen3.5-9b-r":"Qwen3.5-9B-Reasoning",
}
def normalize_model(v): return _MMAP.get(safe_text(v).lower(), safe_text(v))
def model_family(m):
    if "Command" in m: return "Command"
    if m in {"Global","Fire","Earth","Water"}: return "Tiny-Aya"
    if m.startswith("Gemma"): return "Gemma-4"
    if m.startswith("Qwen"): return "Qwen3.5"
    return "Other"
def reasoning_mode(m):
    return "reasoning" if m in {"Command-a-Reasoning","Gemma-4-31B-Reasoning","Qwen3.5-9B-Reasoning"} else "non_reasoning"

def model_scope(m): return "regional" if m in {"Fire","Earth","Water"} else "general"
def provider(m):
    fam = model_family(m)
    if fam in {"Command", "Tiny-Aya"}: return "Cohere"
    if fam == "Gemma-4": return "Gemma"
    if fam == "Qwen3.5": return "Qwen"
    return "Other"
PROVIDER = provider
MODEL_FAMILY = model_family

MODEL_SIZE_B = {
    "Global": 1.6, "Fire": 1.6, "Earth": 1.6, "Water": 1.6,
    "Qwen3.5-0.8B": 0.8, "Qwen3.5-2B": 2.0, "Qwen3.5-9B": 9.0, "Qwen3.5-9B-Reasoning": 9.0,
    "Gemma-4-31B": 31.0, "Gemma-4-31B-Reasoning": 31.0,
    "Command-a": 35.0, "Command-a-Reasoning": 35.0
}

def model_size_b(m):
    return MODEL_SIZE_B.get(m, np.nan)

def is_reasoning(m):
    return 1 if reasoning_mode(m) == "reasoning" else 0
IS_REASONING = is_reasoning

def holm(p_values):
    """Wrapper for Holm-Bonferroni correction."""
    p_values = np.array(p_values)
    mask = ~np.isnan(p_values)
    corrected = np.full(p_values.shape, np.nan)
    if mask.any():
        corrected[mask] = multipletests(p_values[mask], method="holm")[1]
    return corrected

def add_meta(df):
    o = df.copy()
    o["reasoning_mode"] = o["model"].map(reasoning_mode)
    o["is_reasoning"] = o["model"].map(is_reasoning)
    o["model_family"] = o["model"].map(model_family)
    o["provider"] = o["model"].map(provider)
    o["model_size_b"] = o["model"].map(model_size_b)
    o["model_scope"] = o["model"].map(model_scope)
    o["model_order"] = o["model"].map({m:i for i,m in enumerate(MODEL_ORDER)}).fillna(999).astype(int)
    o["condition_label"] = o["context_type"]+"_"+o["conflict_type"]
    return o

def sort_df(df, cols=None):
    if df is None or df.empty: return df
    cols = cols or ["country","model_order","model","lang_condition","context_type","conflict_type","scenario_id"]
    sc = [c for c in cols if c in df.columns]
    return df.sort_values(sc).reset_index(drop=True) if sc else df.reset_index(drop=True)

def load_runs(paths, workdir):
    RAW_MAP = {"model_label":"model","parsed_label":"prediction","is_correct":"accuracy"}
    frames = [pd.read_csv(p).assign(source_file=str(p.relative_to(workdir))) for p in paths]
    runs = pd.concat(frames, ignore_index=True).rename(columns=RAW_MAP)
    for c in ["country","model","model_id","language","context_type","conflict_type"]:
        runs[c] = runs[c].map(safe_text)
    runs["model"] = runs["model"].map(normalize_model)
    runs["scenario_id"] = runs["scenario_id"].astype(str).str.strip()
    for c in ["language","context_type","conflict_type"]: runs[c] = runs[c].str.upper()
    # Rename 'language' → 'lang_condition' and add 'actual_language'
    runs = runs.rename(columns={"language": "lang_condition"})
    runs["actual_language"] = runs.apply(lambda r: resolve_language(r["country"], r["lang_condition"]), axis=1)
    runs["prediction"] = runs["prediction"].map(normalize_label)
    runs["gold_label"] = runs["gold_label"].map(normalize_label)
    for c in ["run_idx","seed","avg_logprob","accuracy","reasoning_length"]:
        if c not in runs.columns: runs[c] = np.nan
        runs[c] = pd.to_numeric(runs[c], errors="coerce")
    for c in ["prompt","raw_response","reasoning_trace"]:
        if c not in runs.columns: runs[c] = ""
    runs["confidence_pct"] = np.exp(runs["avg_logprob"])*100
    runs["parsed"] = runs["prediction"].notna()
    return add_meta(sort_df(runs))

def majority_label(s):
    c = s.dropna().map(normalize_label).dropna()
    if c.empty: return np.nan
    vc = c.value_counts(); w = vc[vc==vc.max()].index.tolist()
    return sorted(w, key=lambda l: LABEL_RANK.get(l,999))[0]

def aggregate(runs):
    recs = []
    for key, g in runs.groupby(CONDITION_KEY, dropna=False, sort=False):
        r = dict(zip(CONDITION_KEY, key))
        pred, gold = majority_label(g["prediction"]), majority_label(g["gold_label"])
        nr = int(g["run_idx"].nunique()) if g["run_idx"].notna().any() else len(g)
        pa = g["prediction"].dropna()
        na = int((pa==pred).sum()) if pd.notna(pred) else 0
        r.update({"prediction":pred,"gold_label":gold,
            "accuracy": float(pred==gold) if pd.notna(pred) and pd.notna(gold) else np.nan,
            "n_runs":nr,"parse_rate":g["prediction"].notna().mean(),
            "agreement_rate": na/nr if nr else np.nan,
            "avg_logprob":g["avg_logprob"].mean(),"confidence_pct":g["confidence_pct"].mean(),
            "reasoning_length_mean":g["reasoning_length"].mean(),
            "source_file":"; ".join(sorted(g["source_file"].dropna().astype(str).unique()))})
        recs.append(r)
    return add_meta(pd.DataFrame(recs))

def wilson_ci(k,n,z=1.96):
    if n<=0: return np.nan,np.nan
    p=k/n; d=1+z**2/n; c=(p+z**2/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z**2/(4*n**2))/d
    return max(0,c-h),min(1,c+h)

def _flat(cols):
    out = []
    for c in cols:
        if isinstance(c, tuple):
            a, b = c
            out.append(f"{a}__{b}" if b else str(a))
        else:
            out.append(str(c))
    return out

def build_pairs(frame, idx, compare, left, right, vcols, filters=None):
    s = frame.copy()
    if filters:
        for c,v in filters.items(): s = s[s[c].isin(v if isinstance(v,(list,tuple,set)) else [v])]
    s = s[s[compare].isin([left,right])]
    w = s[idx+[compare]+vcols].pivot(index=idx, columns=compare, values=vcols).reset_index()
    w.columns = _flat(w.columns)
    la_c, ra_c = f"accuracy__{left}", f"accuracy__{right}"
    w["left_label"],w["right_label"] = left,right
    w["left_prediction"] = w.get(f"prediction__{left}", np.nan)
    w["right_prediction"] = w.get(f"prediction__{right}", np.nan)
    w["left_accuracy"] = pd.to_numeric(w.get(la_c, np.nan), errors="coerce")
    w["right_accuracy"] = pd.to_numeric(w.get(ra_c, np.nan), errors="coerce")
    w["pair_available"] = w["left_prediction"].notna() & w["right_prediction"].notna()
    w["flip"] = np.where(w["pair_available"], (w["left_prediction"]!=w["right_prediction"]).astype(int), np.nan)
    w["accuracy_delta"] = w["right_accuracy"]-w["left_accuracy"]
    w["accuracy_drop"] = np.where(w["pair_available"], ((w["left_accuracy"]==1)&(w["right_accuracy"]==0)).astype(int), np.nan)
    w["accuracy_gain"] = np.where(w["pair_available"], ((w["left_accuracy"]==0)&(w["right_accuracy"]==1)).astype(int), np.nan)
    conds = [(w["left_accuracy"]==1)&(w["right_accuracy"]==1),(w["left_accuracy"]==1)&(w["right_accuracy"]==0),
             (w["left_accuracy"]==0)&(w["right_accuracy"]==1),(w["left_accuracy"]==0)&(w["right_accuracy"]==0)]
    w["transition"] = np.select(conds, ["stable_correct","c2w","w2c","stable_wrong"], "missing")
    if "model" in w.columns:
        w["reasoning_mode"]=w["model"].map(reasoning_mode); w["model_family"]=w["model"].map(model_family)
    return sort_df(w)

def _mcnemar_p(l,r):
    f=pd.DataFrame({"l":l,"r":r}).dropna().astype(int)
    if f.empty: return np.nan
    lo=int(((f.l==1)&(f.r==0)).sum()); ro=int(((f.l==0)&(f.r==1)).sum())
    d=lo+ro
    if d==0: return 1.0
    bp=int(((f.l==1)&(f.r==1)).sum()); bn=int(((f.l==0)&(f.r==0)).sum())
    return mcnemar(np.array([[bp,lo],[ro,bn]]), exact=d<25, correction=d>=25).pvalue

def summarize_pairs(pdf, gcols=None):
    gcols = gcols or []
    c = pdf[pdf["pair_available"]].copy()
    if c.empty: return pd.DataFrame()
    def _one(g):
        n=len(g); nf=int(g["flip"].sum()); lo,hi=wilson_ci(nf,n)
        return pd.Series({"n_pairs":n,"n_flips":nf,"flip_rate":g["flip"].mean(),
            "left_acc":g["left_accuracy"].mean(),"right_acc":g["right_accuracy"].mean(),
            "acc_delta":g["accuracy_delta"].mean(),
            "c2w_rate":g["accuracy_drop"].mean(),"w2c_rate":g["accuracy_gain"].mean(),
            "net_harmful":g["accuracy_drop"].mean()-g["accuracy_gain"].mean(),
            "mcnemar_p":_mcnemar_p(g["left_accuracy"],g["right_accuracy"])})
    if gcols: return c.groupby(gcols, dropna=False).apply(_one).reset_index()
    return _one(c).to_frame().T

def adjust_p(df, col, method="holm"):
    o=df.copy(); ac=f"{col}_{method}"; o[ac]=np.nan
    m=o[col].notna()
    if m.any(): o.loc[m,ac]=multipletests(o.loc[m,col], method=method)[1]
    return o

def macro_f1(y_true, y_pred, labels=None):
    """Compute macro-averaged F1 across classes.

    For each label, compute precision = TP/(TP+FP), recall = TP/(TP+FN),
    F1 = 2*P*R/(P+R).  Macro F1 is the unweighted mean of per-class F1 scores.

    Parameters
    ----------
    y_true, y_pred : array-like of str
        Gold and predicted labels (e.g. 'yes', 'no', 'neutral').
    labels : list of str, optional
        Classes to include.  Defaults to LABEL_ORDER ('yes','no','neutral').

    Returns
    -------
    float   Macro F1 in [0, 1].
    """
    if labels is None:
        labels = LABEL_ORDER
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1s = []
    for lbl in labels:
        tp = int(((y_true == lbl) & (y_pred == lbl)).sum())
        fp = int(((y_true != lbl) & (y_pred == lbl)).sum())
        fn = int(((y_true == lbl) & (y_pred != lbl)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))

