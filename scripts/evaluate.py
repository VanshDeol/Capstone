import json
import re
import gc
import random
import transformers

from pathlib import Path
from collections import Counter
from datetime import datetime

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

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

print(f"Using device: {device}")

# ------------------------------------------------
# PATHS
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

MODEL_DIR = BASE_DIR / "models"

RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------
# RUN DIRECTORY
# ------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

RUN_NAME = f"seed_{SEED}_exposure_{EXPOSURE_SIZE}_timestamp_{timestamp}"

RUN_DIR = RESULTS_DIR / RUN_NAME

RUN_DIR.mkdir(exist_ok=True)

print(f"Run directory: {RUN_DIR}")

# ------------------------------------------------
# LOAD TEST SET
# ------------------------------------------------

with open(DATA_DIR / "openbookqa_final_test.json") as f:

    test_data = json.load(f)

# IMPORTANT
random.shuffle(test_data)

print(f"Loaded {len(test_data)} test examples")

# ------------------------------------------------
# MODELS
# ------------------------------------------------

models = {

    "base_model": None,

    "input_only": MODEL_DIR / f"seed_{SEED}_exposure_{EXPOSURE_SIZE}_finetuned_input_only",

    "input_output": MODEL_DIR / f"seed_{SEED}_exposure_{EXPOSURE_SIZE}_finetuned_input_output"
}

# ------------------------------------------------
# STORAGE
# ------------------------------------------------

results = {}

all_predictions = {}

model_predictions = {}

invalid_prediction_counts = {}

# ------------------------------------------------
# FORMAT CHOICES
# ------------------------------------------------

def format_choices(choices):

    labels = choices["label"]

    texts = choices["text"]

    formatted = []

    for l, t in zip(labels, texts):

        formatted.append(f"{l}. {t}")

    return "\n".join(formatted)

# ------------------------------------------------
# EXTRACT CHOICE
# ------------------------------------------------

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
# LOAD TOKENIZER
# ------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token

# ------------------------------------------------
# EVALUATION LOOP
# ------------------------------------------------

for model_name, adapter_path in models.items():

    print("\n===================================")

    print(f"Evaluating: {model_name}")

    print("===================================")

    # --------------------------------------------
    # DTYPE
    # --------------------------------------------

    dtype = torch.float16 if device != "cpu" else torch.float32

    if (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        dtype = torch.bfloat16

    # --------------------------------------------
    # LOAD BASE MODEL
    # --------------------------------------------

    base_model = AutoModelForCausalLM.from_pretrained(

        BASE_MODEL_NAME,

        torch_dtype=dtype
    )

    base_model.to(device)

    # Remove conflicting generation limit
    base_model.generation_config.max_length = None

    # --------------------------------------------
    # LOAD LORA
    # --------------------------------------------

    if adapter_path is not None:

        model = PeftModel.from_pretrained(
            base_model,
            adapter_path
        )

    else:

        model = base_model

    model.eval()

    correct = 0

    total = 0

    invalid_predictions = 0

    predictions_for_model = []

    # --------------------------------------------
    # EVALUATE
    # --------------------------------------------

    for idx, item in enumerate(test_data[:EVAL_SIZE]):

        question = item["question_stem"]

        choices = format_choices(item["choices"])

        gold_answer = item["answerKey"]

        prompt = f"""
Question:
{question}

Choices:
{choices}

Answer with only A, B, C, or D.

Answer:
"""

        inputs = tokenizer(

            prompt,

            return_tensors="pt"

        ).to(device)

        with torch.no_grad():

            outputs = model.generate(

                **inputs,

                max_new_tokens=5,

                do_sample=False,

                pad_token_id=tokenizer.pad_token_id
            )

        # IMPORTANT:
        # ONLY GENERATED TOKENS
        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        generated_text = tokenizer.decode(

            generated_tokens,

            skip_special_tokens=True
        )

        prediction = generated_text.strip()

        predicted_choice = extract_choice(
            prediction
        )

        if predicted_choice is None:

            invalid_predictions += 1

        is_correct = (

            predicted_choice is not None

            and predicted_choice == gold_answer
        )

        if is_correct:

            correct += 1

        total += 1

        prediction_record = {

            "question_id": idx,

            "question": question,

            "choices": choices,

            "gold_choice": gold_answer,

            "prediction_text": prediction,

            "predicted_choice": predicted_choice,

            "correct": is_correct
        }

        predictions_for_model.append(
            prediction_record
        )

        if idx % 50 == 0:

            print(f"{model_name} | {idx}/{EVAL_SIZE}")

    accuracy = correct / total

    results[model_name] = accuracy

    invalid_prediction_counts[model_name] = invalid_predictions

    model_predictions[model_name] = predictions_for_model

    all_predictions[model_name] = predictions_for_model

    # --------------------------------------------
    # CLEANUP MEMORY
    # --------------------------------------------

    del model

    del base_model

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

# ------------------------------------------------
# TRANSITION ANALYSIS
# ------------------------------------------------

transition_analysis = {}

base_preds = model_predictions["base_model"]

for compare_model in ["input_only", "input_output"]:

    compare_preds = model_predictions[compare_model]

    transitions = Counter()

    for base_item, compare_item in zip(
        base_preds,
        compare_preds
    ):

        base_correct = base_item["correct"]

        compare_correct = compare_item["correct"]

        if not base_correct and compare_correct:

            transitions["wrong_to_correct"] += 1

        elif base_correct and not compare_correct:

            transitions["correct_to_wrong"] += 1

        elif base_correct and compare_correct:

            transitions["correct_to_correct"] += 1

        else:

            transitions["wrong_to_wrong"] += 1

    transition_analysis[compare_model] = dict(
        transitions
    )

# ------------------------------------------------
# AGREEMENT ANALYSIS
# ------------------------------------------------

agreement_analysis = {}

for compare_model in ["input_only", "input_output"]:

    compare_preds = model_predictions[compare_model]

    agreement_count = 0

    for base_item, compare_item in zip(
        base_preds,
        compare_preds
    ):

        if (

            base_item["predicted_choice"]

            == compare_item["predicted_choice"]
        ):

            agreement_count += 1

    agreement = agreement_count / len(
        base_preds
    )

    agreement_analysis[compare_model] = agreement

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

config = {

    "seed": SEED,

    "model": BASE_MODEL_NAME,

    "benchmark": "OpenBookQA",

    "eval_size": EVAL_SIZE,

    "device": device,

    "torch_version": torch.__version__,

    "transformers_version": transformers.__version__
}

# ------------------------------------------------
# SAVE FILES
# ------------------------------------------------

with open(RUN_DIR / "accuracy.json", "w") as f:

    json.dump(results, f, indent=4)

with open(RUN_DIR / "predictions.json", "w") as f:

    json.dump(all_predictions, f, indent=4)

with open(RUN_DIR / "transitions.json", "w") as f:

    json.dump(transition_analysis, f, indent=4)

with open(RUN_DIR / "agreement.json", "w") as f:

    json.dump(agreement_analysis, f, indent=4)

with open(RUN_DIR / "invalid_predictions.json", "w") as f:

    json.dump(invalid_prediction_counts, f, indent=4)

with open(RUN_DIR / "config.json", "w") as f:

    json.dump(config, f, indent=4)

# ------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------

print("\n===================================")

print("FINAL ACCURACY")

print("===================================")

for model_name, accuracy in results.items():

    print(f"{model_name}: {accuracy:.4f}")

print("\n===================================")

print("INVALID PREDICTIONS")

print("===================================")

for model_name, count in invalid_prediction_counts.items():

    print(f"{model_name}: {count}")

print("\n===================================")

print("TRANSITION ANALYSIS")

print("===================================")

for model_name, transitions in transition_analysis.items():

    print(f"\n{model_name}")

    for k, v in transitions.items():

        print(f"{k}: {v}")

print("\n===================================")

print("AGREEMENT ANALYSIS")

print("===================================")

for model_name, agreement in agreement_analysis.items():

    print(f"{model_name}: {agreement:.4f}")

print(f"\nAll results saved to: {RUN_DIR}")