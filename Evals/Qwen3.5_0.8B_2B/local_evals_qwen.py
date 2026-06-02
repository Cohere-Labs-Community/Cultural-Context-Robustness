"""
local_evals_qwen.py -- Local evaluation script for Qwen3.5 family - 0.8B and 2B using HuggingFace Transformers.
Handles CSV processing, logprob extraction, and reasoning trace capture.

================================================================================
SETUP
================================================================================

-- Install dependencies --------------------------------------------------------
    pip install torch transformers accelerate pandas numpy

    # Apple Silicon (M1/M2/M3) -- MPS backend is used automatically.
    # Make sure you have torch >= 2.0 with MPS support:
    pip install torch --index-url https://download.pytorch.org/whl/cpu

-- Model download ---------------------------------------------------------------
    The model is downloaded automatically from HuggingFace on first run.
    Cached at: ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B
    Size on disk: ~1.6 GB (bfloat16)

    To pre-download manually:
    python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
               AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B'); \
               AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-0.8B', torch_dtype='auto')"

================================================================================
USAGE
================================================================================

-- Basic run (no reasoning) ----------------------------------------------------
    python local_evals_qwen.py \
        --input  data/normad_india_template.csv \
        --output results/normad_india_qwen0.8b.csv

-- With reasoning mode (enables Qwen3.5 <think> tags via tokenizer) ------------
    python local_evals_qwen.py \
        --input  data/normad_india_template.csv \
        --output results/normad_india_qwen0.8b_reasoning.csv \
        --reasoning

-- Different model --------------------------------------------------------------
    python local_evals_qwen.py \
        --input  data/normad_india_template.csv \
        --output results/normad_india_qwen2b.csv \
        --model  Qwen/Qwen3.5-2B

-- Custom checkpoint interval (default: every 5 rows) --------------------------
    python local_evals_qwen.py \
        --input  data/normad_india_template.csv \
        --output results/normad_india_qwen0.8b.csv \
        --checkpoint 3

-- Run all countries in a loop (bash) ------------------------------------------
    for country in vietnam turkey; do
        python local_evals_qwen.py \
            --input  data/normad_${country}_template.csv \
            --output results/normad_${country}_qwen0.8b.csv
    done

================================================================================
OUTPUT COLUMNS (appended to input template columns)
================================================================================
    raw_response    : Full model output text
    parsed_label    : Extracted Yes / No / Neither (or NaN on parse failure)
    avg_logprob     : Mean token log-probability over generated tokens
    reasoning_trace : Content from <think> tags (empty if reasoning mode off)
    elapsed_s       : Inference wall-clock time in seconds
    is_correct      : Boolean -- parsed_label matches gold_label

================================================================================
HOW THINKING MODE IS CONTROLLED
================================================================================
    Uses tokenizer.apply_chat_template(..., enable_thinking=True/False) --
    the official HuggingFace mechanism for Qwen3.5. This cleanly replaces
    all the llama.cpp chat_template_kwargs workarounds.

    No thinking: tokenizer injects <think>\\n\\n</think>\\n in the prompt.
    Thinking on: tokenizer leaves the assistant turn open for <think> output.
================================================================================
"""
from __future__ import annotations

import time
import logging
import re
import pandas as pd
import numpy as np
from typing import Optional
import argparse

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
import math

# -- Device detection ----------------------------------------------------------

def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# -- Singleton model + tokenizer (lazy-loaded) ---------------------------------

_model     = None
_tokenizer = None
_current_model_id = None


def _get_model_and_tokenizer(model_id: str):
    """Lazy-load the model and tokenizer, swapping if model_id changes."""
    global _model, _tokenizer, _current_model_id

    if _model is not None and _current_model_id == model_id:
        return _model, _tokenizer

    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if _model is not None:
        del _model
        del _tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    device = _get_device()
    print(f"[model] Loading {model_id} on {device} ...")

    _tokenizer = AutoTokenizer.from_pretrained(model_id)

    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,   # bfloat16 on all devices; ~1.6 GB for 0.8B
        device_map=device,
    )
    _model.eval()

    _current_model_id = model_id
    print(f"[model] Ready.")
    return _model, _tokenizer


# -- Inference -----------------------------------------------------------------

def generate_transformers(
    model_id: str,
    messages: list[dict[str, str]],
    temperature: float = 0.1,
    top_p: float = 0.75,
    max_new_tokens: int = 256,
    reasoning_mode: bool = False,
) -> tuple[str, float, float, Optional[str]]:
    """
    Run inference and return (response_text, avg_logprob, elapsed_s, error).
    """
    try:
        model, tokenizer = _get_model_and_tokenizer(model_id)
        device = next(model.parameters()).device

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=reasoning_mode,
        )

        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1]

        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "repetition_penalty": 1.1,
            "pad_token_id": tokenizer.eos_token_id,
            "return_dict_in_generate": True,
        }

        # Request unwarped logits to prevent 0.0 logprobs without crashing memory
        import transformers
        hf_version = tuple(map(int, transformers.__version__.split('.')[:2]))
        if hf_version >= (4, 38):
            generate_kwargs["output_logits"] = True
        else:
            generate_kwargs["output_scores"] = True

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)
        elapsed = time.perf_counter() - t0

        # Decode only the newly generated tokens
        generated_ids = outputs.sequences[0, input_length:]
        content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Compute avg log-probability token-by-token (Memory-Safe for Mac/MPS)
        avg_logprob = 0.0
        if len(generated_ids) > 0:
            # Grab whichever raw distributions the model generated
            raw_logits_tuple = getattr(outputs, "logits", None) or getattr(outputs, "scores", None)
            
            if raw_logits_tuple is not None:
                logprob_sum = 0.0
                
                # Process step-by-step to prevent Mac OOM crashes
                for i, step_logits in enumerate(raw_logits_tuple):
                    # Fail-safe bound check
                    if i >= len(generated_ids): 
                        break
                        
                    token_id = generated_ids[i].item()
                    
                    # step_logits is (1, vocab_size). Squeeze it and cast to float.
                    # Casting a 1D tensor to float is virtually free in RAM.
                    step_logits_fp32 = step_logits.squeeze(0).float()
                    
                    # Apply log_softmax to just this single step
                    step_log_probs = F.log_softmax(step_logits_fp32, dim=-1)
                    
                    # Extract logprob of the specific token the model chose
                    logprob_sum += step_log_probs[token_id].item()
                    
                # avg_logprob = logprob_sum / len(generated_ids)
                avg_logprob = math.exp(logprob_sum)

        return content, avg_logprob, elapsed, None

    except Exception as exc:
        logger.error(f"Transformers inference error: {exc}")
        return "", 0.0, 0.0, str(exc)


# -- Parsing -------------------------------------------------------------------

def parse_label(response_text: str) -> str:
    """
    Extract Yes, No, or Neither from the response.
    Strips <think>...</think> blocks before parsing so reasoning text
    never contaminates label extraction.
    """
    clean_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).lower()

    patterns = [
        r'answer\s*[:\-]?\s*(yes|no|neither)',
        r'\b(yes|no|neither)\b',
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, clean_text))
        if matches:
            return matches[-1].group(1).capitalize()
    return "NaN"


def extract_reasoning(response_text: str) -> str:
    """
    Extract reasoning trace from Qwen3.5's <think> tags.
    Falls back to text before the last 'answer' keyword if tags are absent.
    """
    match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    answer_idx = response_text.lower().rfind("answer")
    if answer_idx != -1:
        return response_text[:answer_idx].strip()

    return ""


# -- Main Execution Logic ------------------------------------------------------

def run_eval(
    input_csv: str,
    output_csv: str,
    model_id: str,
    reasoning_mode: bool = False,
    checkpoint_every: int = 5,
):
    df = pd.read_csv(input_csv)
    print(f"Loaded      : {len(df)} rows from {input_csv}")
    print(f"Model       : {model_id}")
    print(f"Device      : {_get_device()}")
    print(f"Reasoning   : {'ON  -- enable_thinking=True' if reasoning_mode else 'OFF -- enable_thinking=False'}")
    print(f"Checkpoint  : every {checkpoint_every} rows")
    print("-" * 60)

    df['model_label'] = model_id

    TEMP           = 0.1
    TOP_P          = 0.75
    MAX_NEW_TOKENS = 1024 
    REASONING_EXTRA = 1024

    max_new_tokens = MAX_NEW_TOKENS + (REASONING_EXTRA if reasoning_mode else 0)

    results = []

    for idx, row in df.iterrows():
        print(f"Row {idx + 1}/{len(df)} | Scenario {row['scenario_id']}")

        messages = [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": row['prompt']},
        ]

        raw_response, avg_logprob, elapsed, error = generate_transformers(
            model_id=model_id,
            messages=messages,
            temperature=TEMP,
            top_p=TOP_P,
            max_new_tokens=max_new_tokens,
            reasoning_mode=reasoning_mode,
        )

        result = row.to_dict()

        if error:
            print(f"  X Error: {error}")
            result['raw_response']    = f"ERROR: {error}"
            result['parsed_label']    = "NaN"
            result['avg_logprob']     = 0.0
            result['reasoning_trace'] = ""
            result['elapsed_s']       = 0.0
            result['is_correct']      = False
        else:
            parsed = parse_label(raw_response)
            gold   = str(row['gold_label']).strip().lower()

            result['raw_response']    = raw_response
            result['parsed_label']    = parsed
            result['avg_logprob']     = avg_logprob
            result['reasoning_trace'] = extract_reasoning(raw_response)
            result['elapsed_s']       = round(elapsed, 3)
            result['is_correct']      = (gold == parsed.lower())

            status = "OK" if result['is_correct'] else "!!"
            print(f"  [{status}] Parsed: {parsed:8s} | Gold: {str(row['gold_label']):8s} | logp: {avg_logprob:.3f} | {elapsed:.1f}s")

        results.append(result)

        if (idx + 1) % checkpoint_every == 0:
            pd.DataFrame(results).to_csv(output_csv, index=False)
            print(f"  [checkpoint saved -> {output_csv}]")

    final_df = pd.DataFrame(results)
    final_df.to_csv(output_csv, index=False)

    n_total   = len(final_df)
    n_correct = final_df['is_correct'].sum()
    n_nan     = (final_df['parsed_label'] == 'NaN').sum()
    valid_lp  = final_df.loc[final_df['parsed_label'].ne('NaN'), 'avg_logprob']

    print("\n" + "=" * 60)
    print(f"DONE  ->  {output_csv}")
    print(f"  Accuracy    : {n_correct / n_total:.3f}  ({n_correct}/{n_total})")
    print(f"  NaN rate    : {n_nan}/{n_total} rows failed to parse")
    print(f"  Avg logprob : {valid_lp.mean():.4f}  (valid rows only)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NormAd-style local evals for Qwen3.5-0.8B via HuggingFace Transformers."
    )
    parser.add_argument("--input",      type=str, required=True,
                        help="Path to input CSV template.")
    parser.add_argument("--output",     type=str, required=True,
                        help="Path to save output CSV.")
    parser.add_argument("--model",      type=str, default="Qwen/Qwen3.5-0.8B",
                        help="HuggingFace model ID (default: Qwen/Qwen3.5-0.8B).")
    parser.add_argument("--reasoning",  action="store_true",
                        help="Enable reasoning mode (enable_thinking=True).")
    parser.add_argument("--checkpoint", type=int, default=5,
                        help="Save progress every N rows (default: 5).")

    args = parser.parse_args()

    run_eval(
        input_csv=args.input,
        output_csv=args.output,
        model_id=args.model,
        reasoning_mode=args.reasoning,
        checkpoint_every=args.checkpoint,
    )