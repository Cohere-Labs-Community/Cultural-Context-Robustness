# What we do: We collect all unique raw model responses, send them to the Cohere API to be parsed
# by an LLM into clean decisions ('yes', 'no', 'neutral'), and save them to 'final_results/'.
# Concurrency Logic:
# 1. Maps raw responses to unique IDs and groups them into batches of 10 to minimize API cost and latency.
# 2. Distributes batch tasks across multiple client instances (API keys) in a round-robin cycle.
# 3. Uses client-specific semaphores to cap maximum concurrency at 10 requests per API key.
# 4. Executes robust retry logic with exponential backoff on JSON parse or connection exceptions.

import pandas as pd
import asyncio
from cohere import AsyncClientV2
from tqdm.asyncio import tqdm
import json
import logging
import os

logging.basicConfig(level=logging.INFO)

KEYS = []##list of keys]
INPUT_DIR = "merged_raw_results"
OUTPUT_DIR = "final_results"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = [
    "aggregated_all_models_india.csv",
    "aggregated_all_models_turkey.csv",
    "aggregated_all_models_vietnam.csv"
]

# UPDATED PROMPT: Strict JSON schema enforced
PROMPT_TEMPLATE = """You are an expert at extracting the final decision from a model's response.
I will provide you with a list of responses, each prefixed with an ID.
For each response, extract the final decision.
The label should be EXACTLY one of: "yes", "no", "neutral", or "unparseable".

Rules:
- Any version of "yes" (e.g., 1) Yes, yes, Yes, はい, etc) should be "yes".
- Any version of "no" (e.g., 2) No, no, No, いいえ, etc) should be "no".
- Any version of "neither" or "neutral" (e.g., 3) Neither, neither, neutral, どちらでもない, etc) should be "neutral".
- If it's unparseable, empty, or you really cannot determine, return "unparseable".

Output the results strictly as a JSON object where the keys exactly match the IDs provided (e.g., "ID_0") and values are the extracted labels. Do not output any nested objects, conversational text, or markdown formatting outside the JSON.

Example format:
{{
  "ID_0": "yes",
  "ID_1": "no"
}}

Responses to parse:
{batch_text}
"""

CONCURRENCY_PER_KEY = 10
semaphore_per_key = [asyncio.Semaphore(CONCURRENCY_PER_KEY) for _ in KEYS]
clients = [AsyncClientV2(key) for key in KEYS]

CACHE_FILE = "label_cache_batch.json"
cache = {}

async def parse_batch(idx, batch_items):
    # batch_items is a list of tuples: (id, raw_response)
    batch_text = ""
    for b_id, rr in batch_items:
        batch_text += f"ID_{b_id}:\n{rr}\n\n---\n\n"
        
    client_idx = idx % len(KEYS)
    client = clients[client_idx]
    sem = semaphore_per_key[client_idx]
    
    async with sem:
        retries = 3
        for attempt in range(retries):
            try:
                res = await client.chat(
                    model="command-a-03-2025",
                    messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(batch_text=batch_text)}],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                
                content = res.message.content[0].text.strip()
                
                # Strip Markdown formatting if the model still includes it
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()
                    
                parsed_json = json.loads(content)
                
                # FIX 1: Handle cases where the LLM nests the output inside a parent key (like "results")
                if len(parsed_json) == 1 and isinstance(list(parsed_json.values())[0], dict):
                    parsed_json = list(parsed_json.values())[0]
                
                results = {}
                for b_id, _ in batch_items:
                    # FIX 2: Look for multiple variations of the ID key
                    possible_keys = [f"ID_{b_id}", f"id_{b_id}", str(b_id), f"ID {b_id}", int(b_id)]
                    raw_label = None
                    
                    for k in possible_keys:
                        if k in parsed_json:
                            raw_label = parsed_json[k]
                            break
                            
                    if raw_label is None:
                        logging.warning(f"Batch {idx}: Could not find ID {b_id} in JSON keys: {list(parsed_json.keys())}")
                        raw_label = "unparseable"

                    # Normalize the label
                    label = str(raw_label).lower().strip()
                    if "yes" in label and len(label) < 15: label = "yes"
                    elif "no" in label and len(label) < 15: label = "no"
                    elif "neutral" in label and len(label) < 15: label = "neutral"
                    else: label = "unparseable"
                    
                    results[b_id] = label
                    
                return results
                
            except json.JSONDecodeError as je:
                # FIX 3: Catch JSON errors specifically and log the raw output for debugging
                logging.error(f"JSON Parsing Error on batch {idx}, attempt {attempt}: {je}\nRaw LLM Output:\n{content}")
                if attempt == retries - 1:
                    return {b_id: "unparseable" for b_id, _ in batch_items}
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                logging.error(f"Failed to parse batch {idx} on attempt {attempt}: {type(e).__name__}: {e}")
                if attempt == retries - 1:
                    return {b_id: "unparseable" for b_id, _ in batch_items}
                await asyncio.sleep(2 ** attempt)

async def process_all():
    dfs = {}
    unique_raw_responses = set()
    
    # Load files
    for f in FILES:
        filepath = os.path.join(INPUT_DIR, f)
        if not os.path.exists(filepath):
            logging.error(f"File not found: {filepath}. Skipping.")
            continue
            
        df = pd.read_csv(filepath)
        dfs[f] = df
        for rr in df["raw_response"].unique():
            if isinstance(rr, str) and rr.strip():
                unique_raw_responses.add(rr)
                
    logging.info(f"Total unique raw_responses across all valid files: {len(unique_raw_responses)}")
    
    # Load cache if it exists to avoid re-running successful parses
    global cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
            logging.info(f"Loaded {len(cache)} items from cache.")
        except Exception as e:
            logging.warning(f"Could not load cache: {e}")

    to_process = []
    # Map raw_response to a unique numeric ID
    rr_to_id = {}
    id_to_rr = {}
    for i, rr in enumerate(unique_raw_responses):
        rr_to_id[rr] = i
        id_to_rr[i] = rr
        
        # Only process if it's not in the cache OR if it was previously cached as "unparseable"
        if rr not in cache or cache[rr] == "unparseable":
            to_process.append((i, rr))
            
    logging.info(f"To process via API (new or previously unparseable): {len(to_process)}")
    
    BATCH_SIZE = 10
    batches = [to_process[i:i + BATCH_SIZE] for i in range(0, len(to_process), BATCH_SIZE)]
    logging.info(f"Total batches to run: {len(batches)}")
    
    tasks = []
    for i, batch in enumerate(batches):
        tasks.append(parse_batch(i, batch))
        
    if tasks:
        results = await tqdm.gather(*tasks)
        
        for batch_res in results:
            for b_id, label in batch_res.items():
                rr = id_to_rr[b_id]
                cache[rr] = label
                
        # Save updated cache
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
            logging.info(f"Saved {len(cache)} results to {CACHE_FILE}")
            
    # Apply to dataframes and save
    for f in dfs:
        df = dfs[f]
        
        def map_label(rr):
            if not isinstance(rr, str) or not rr.strip():
                return "unparseable"
            return cache.get(rr, "unparseable")
            
        df["parsed_label"] = df["raw_response"].apply(map_label)
        
        out_f = f.replace(".csv", "_llm_parsed_fixed.csv")
        out_path = os.path.join(OUTPUT_DIR, out_f)
        df.to_csv(out_path, index=False)
        logging.info(f"Saved parsed results to {out_path}")

    # Print summary
    if dfs:
        combined_df = pd.concat(dfs.values())
        summary = combined_df['parsed_label'].value_counts()
        print("\n" + "="*40)
        print("FINAL LABEL SUMMARY (ALL FILES)")
        print("="*40)
        print(summary)
        print("="*40)

if __name__ == "__main__":
    asyncio.run(process_all())