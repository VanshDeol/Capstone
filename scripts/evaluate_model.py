import json
import re
import gc
import argparse
import random
import sys
import transformers
from pathlib import Path
from collections import Counter

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed
)
from peft import PeftModel

# ------------------------------------------------
# PATHS & IMPORTS
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from models import MODELS

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--exposure_size", type=int, required=True)
parser.add_argument("--condition", type=str, required=True)
args = parser.parse_args()

SEED = args.seed
EXPOSURE_SIZE = args.exposure_size
CONDITION = args.condition
MODEL_KEY = args.model

MODEL_CONFIG = MODELS[MODEL_KEY]
BASE_MODEL_NAME = MODEL_CONFIG["hf_name"]

# Fixed evaluation length as requested
EVAL_SIZE = 500

# ------------------------------------------------
# RANDOM SEED
# ------------------------------------------------

set_seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ------------------------------------------------
# DEVICE DETECTION
# ------------------------------------------------

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# ------------------------------------------------
# DYNAMIC PATHS
# ------------------------------------------------

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

ADAPTER_DIR = MODEL_DIR / MODEL_KEY / f"seed_{SEED}" / f"exposure_{EXPOSURE_SIZE}" / CONDITION
RUN_DIR = RESULTS_DIR / MODEL_KEY / f"seed_{SEED}" / f"exposure_{EXPOSURE_SIZE}" / CONDITION
BASE_RUN_DIR = RESULTS_DIR / MODEL_KEY / f"seed_{SEED}" / f"exposure_{EXPOSURE_SIZE}" / "base_model"

RUN_DIR.mkdir(parents=True, exist_ok=True)
BASE_RUN_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------
# LOAD TEST SET
# ------------------------------------------------

with open(DATA_DIR / "openbookqa_final_test.json") as f:
    test_data = json.load(f)

# Isolate random presentation order to this seed
random.shuffle(test_data)

# ------------------------------------------------
# EVALUATION HELPERS
# ------------------------------------------------

def format_choices(choices):
    labels = choices["label"]
    texts = choices["text"]
    formatted = []
    for l, t in zip(labels, texts):
        formatted.append(f"{l}. {t}")
    return "\n".join(formatted)

def extract_choice(text):
    text = text.upper().strip()
    patterns = [
        r"^\s*([ABCD])",
        r"ANSWER:\s*([ABCD])"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None

# ------------------------------------------------
# EXECUTION
# ------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.float16 if device != "cpu" else torch.float32
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16

results = {}
all_predictions = {}

BASE_PREDS_FILE = BASE_RUN_DIR / "predictions.json"

if BASE_PREDS_FILE.exists():
    print("\nFound cached base model predictions. Skipping redundant evaluation...")
    with open(BASE_PREDS_FILE) as f:
        all_predictions["base_model"] = json.load(f)
        
    correct = sum(1 for p in all_predictions["base_model"] if p["correct"])
    results["base_model"] = correct / EVAL_SIZE
    models_to_eval = [(CONDITION, ADAPTER_DIR)]
else:
    models_to_eval = [
        ("base_model", None),
        (CONDITION, ADAPTER_DIR)
    ]

for model_name, adapter_path in models_to_eval:
    print(f"\nEvaluating {model_name}...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=dtype
    )
    base_model.to(device)
    
    # Remove conflicting generation limit
    base_model.generation_config.max_length = None
    
    if adapter_path is not None:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model
        
    model.eval()
    
    correct = 0
    total = 0
    predictions_for_model = []
    
    for idx, item in enumerate(test_data[:EVAL_SIZE]):
        question = item["question_stem"]
        choices = format_choices(item["choices"])
        gold_answer = item["answerKey"]
        
        prompt = f"""Question:
{question}

Choices:
{choices}

Answer with only A, B, C, or D.

Answer:
"""
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        predicted_choice = extract_choice(prediction)
        
        is_correct = (predicted_choice is not None and predicted_choice == gold_answer)
        if is_correct:
            correct += 1
        total += 1
        
        predictions_for_model.append({
            "question_id": idx,
            "gold_choice": gold_answer,
            "prediction_text": prediction,
            "predicted_choice": predicted_choice,
            "correct": is_correct
        })
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{EVAL_SIZE} examples...")
        
    results[model_name] = correct / total
    all_predictions[model_name] = predictions_for_model
    
    del model
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
# Cache the base model predictions if we just generated them
if not BASE_PREDS_FILE.exists() and "base_model" in all_predictions:
    with open(BASE_PREDS_FILE, "w") as f:
        json.dump(all_predictions["base_model"], f, indent=4)

# ------------------------------------------------
# ANALYSIS (Base vs Finetuned)
# ------------------------------------------------

transitions = Counter()
agreement_count = 0

base_preds = all_predictions["base_model"]
fine_preds = all_predictions[CONDITION]

for b, f in zip(base_preds, fine_preds):
    # Transition
    if not b["correct"] and f["correct"]:
        transitions["wrong_to_correct"] += 1
    elif b["correct"] and not f["correct"]:
        transitions["correct_to_wrong"] += 1
    elif b["correct"] and f["correct"]:
        transitions["correct_to_correct"] += 1
    else:
        transitions["wrong_to_wrong"] += 1
        
    # Agreement
    if b["predicted_choice"] == f["predicted_choice"]:
        agreement_count += 1

# ------------------------------------------------
# SAVE EVERYTHING
# ------------------------------------------------

with open(RUN_DIR / "accuracy.json", "w") as f:
    json.dump(results, f, indent=4)
    
with open(RUN_DIR / "predictions.json", "w") as f:
    json.dump(all_predictions, f, indent=4)
    
with open(RUN_DIR / "transitions.json", "w") as f:
    json.dump(dict(transitions), f, indent=4)
    
with open(RUN_DIR / "agreement.json", "w") as f:
    json.dump({"agreement": agreement_count / EVAL_SIZE}, f, indent=4)

print(f"\nAll results saved to: {RUN_DIR}")