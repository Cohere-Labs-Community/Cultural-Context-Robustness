import pandas as pd
from pathlib import Path
import os

COUNTRIES = ["india", "turkey", "vietnam"]

TARGET_COLS = [
    "scenario_id", "country", "model_label", "model_id", "language",
    "context_type", "conflict_type", "run_idx", "seed", "prompt",
    "raw_response", "parsed_label", "avg_logprob", "gold_label",
    "reasoning_trace", "is_correct", "reasoning_length",
]

# Provide mappings for new models to distinguish reasoning vs non-reasoning
LABEL_OVERRIDES = {
    "normad_india_gemma-4-31b-it-multilingual_non_reasoning.csv": "Gemma-4-31B-NR",
    "normad_india_gemma-4-31b-it-multilingual_reasoning.csv": "Gemma-4-31B-R",
    "normad_india_qwen3-5-0.8b.csv": "Qwen3.5-0.8B",
    "normad_india_qwen3-5-2b.csv": "Qwen3.5-2B",
    "normad_india_qwen3-5-9b-aml_non_reasoning.csv": "Qwen3.5-9B-NR",
    "normad_india_qwen3-5-9b-aml_reasoning.csv": "Qwen3.5-9B-R",
    "normad_turkey_gemma-4-31b-it-multilingual_non_reasoning.csv": "Gemma-4-31B-NR",
    "normad_turkey_gemma-4-31b-it-multilingual_reasoning.csv": "Gemma-4-31B-R",
    "normad_turkey_qwen3-5-0.8b.csv": "Qwen3.5-0.8B",
    "normad_turkey_qwen3-5-2b.csv": "Qwen3.5-2B",
    "normad_turkey_qwen3-5-9b-aml_non_reasoning.csv": "Qwen3.5-9B-NR",
    "normad_turkey_qwen3-5-9b-aml_reasoning.csv": "Qwen3.5-9B-R",
    "normad_vietnam_gemma-4-31b-it-multilingual_non_reasoning.csv": "Gemma-4-31B-NR",
    "normad_vietnam_gemma-4-31b-it-multilingual_reasoning.csv": "Gemma-4-31B-R",
    "normad_vietnam_qwen3-5-0.8b.csv": "Qwen3.5-0.8B",
    "normad_vietnam_qwen3-5-2b.csv": "Qwen3.5-2B",
    "normad_vietnam_qwen3-5-9b-aml_non_reasoning.csv": "Qwen3.5-9B-NR",
    "normad_vietnam_qwen3-5-9b-aml_reasoning.csv": "Qwen3.5-9B-R",
}

def normalize_csv(df: pd.DataFrame, fname: str) -> pd.DataFrame:
    df = df.copy()
    
    # Apply override label if applicable to differentiate variants
    if fname in LABEL_OVERRIDES:
        df["model_label"] = LABEL_OVERRIDES[fname]
        
    if "model_id" not in df.columns:
        df["model_id"] = df["model_label"]

    if "elapsed_s" in df.columns and "reasoning_length" not in df.columns:
        if "reasoning_trace" in df.columns:
            df["reasoning_length"] = df["reasoning_trace"].apply(
                lambda x: len(str(x)) if pd.notna(x) and str(x).strip() else 0
            )
        else:
            df["reasoning_length"] = 0

    for col in TARGET_COLS:
        if col not in df.columns:
            df[col] = "" if col in ("prompt", "raw_response", "reasoning_trace") else pd.NA

    return df[TARGET_COLS]

def main():
    base_dir = Path(__file__).parent.parent
    output_dir = Path(__file__).parent / "merged_raw_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for country in COUNTRIES:
        print(f"\nProcessing {country.upper()}...")
        
        # Original core models unaggregated CSV (from Cohere evals folder)
        base_file = base_dir / "Evals" / "Cohere" / f"data_{country}" / f"unaggregated_results_{country}.csv"
        df_base = pd.read_csv(base_file)
        
        # Make sure we only keep core models (in case it was previously modified)
        core_models = {'Global', 'Fire', 'Earth', 'Water', 'Command-a', 'Command-a-Reasoning'}
        df_base = df_base[df_base['model_label'].isin(core_models)]
        df_base = normalize_csv(df_base, base_file.name)
        
        frames = [df_base]
        
        # Read new models for the given country
        new_model_files = [f for f in LABEL_OVERRIDES.keys() if f"_{country}_" in f]
        
        for fname in new_model_files:
            # Determine correct evaluations folder based on model size
            if "qwen3-5-0.8b" in fname or "qwen3-5-2b" in fname:
                fpath = base_dir / "Evals" / "Qwen3.5_0.8B_2B" / "results" / fname
            else:
                fpath = base_dir / "Evals" / "Qwen3.5_9B_Gemma4" / "results" / fname
                
            if not fpath.exists():
                print(f"  WARNING: {fname} not found at {fpath}!")
                continue
            
            df = pd.read_csv(fpath)
            df_norm = normalize_csv(df, fname)
            frames.append(df_norm)
            print(f"  Added {LABEL_OVERRIDES[fname]} ({len(df_norm)} rows) from {fpath.name}")
            
        merged = pd.concat(frames, ignore_index=True)
        
        out_path = output_dir / f"aggregated_all_models_{country}.csv"
        merged.to_csv(out_path, index=False)
        print(f"Saved {country.upper()} to {out_path.name} (Total rows: {len(merged)})")
        print(f"Models included: {sorted(merged['model_label'].unique())}")

if __name__ == "__main__":
    main()

