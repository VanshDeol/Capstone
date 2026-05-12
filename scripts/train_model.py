import json
import argparse
import sys
from pathlib import Path

import torch
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

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
parser.add_argument("--model", type=str, required=True, help="Key from config/models.py (e.g., qwen)")
parser.add_argument("--seed", type=int, required=True, help="Random seed for training")
parser.add_argument("--exposure_size", type=int, required=True, help="Number of exposure examples")
parser.add_argument("--condition", type=str, required=True, help="input_only or input_output")
args = parser.parse_args()

SEED = args.seed
EXPOSURE_SIZE = args.exposure_size
CONDITION = args.condition
MODEL_KEY = args.model

if MODEL_KEY not in MODELS:
    raise ValueError(f"Model {MODEL_KEY} not found in config/models.py")

MODEL_CONFIG = MODELS[MODEL_KEY]
BASE_MODEL_NAME = MODEL_CONFIG["hf_name"]
LORA_TARGETS = MODEL_CONFIG["lora_targets"]

# ------------------------------------------------
# RANDOM SEED
# ------------------------------------------------

set_seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Using seed: {SEED}")

MAX_LENGTH = 256
EPOCHS = 1
BATCH_SIZE = 1
LORA_RANK = 8

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
# DYNAMIC PATHS
# ------------------------------------------------

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_FILE = DATA_DIR / f"openbookqa_{CONDITION}_{EXPOSURE_SIZE}.json"

OUTPUT_DIR = MODEL_DIR / MODEL_KEY / f"seed_{SEED}" / f"exposure_{EXPOSURE_SIZE}" / CONDITION
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

with open(DATA_FILE) as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

# IMPORTANT: Isolate stochastic batch randomness to the seed
dataset = dataset.shuffle(seed=SEED)

print(f"Loaded {len(dataset)} samples from {DATA_FILE.name}")

# ------------------------------------------------
# LOAD TOKENIZER
# ------------------------------------------------

print(f"Loading tokenizer for {BASE_MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# Fix for models that lack a pad token (like Llama, Qwen, Phi)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

print(f"Loading base model {BASE_MODEL_NAME}...")
dtype = torch.float16 if device != "cpu" else torch.float32
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=dtype
)
model.to(device)
model.config.use_cache = False

# ------------------------------------------------
# LORA CONFIG
# ------------------------------------------------

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=LORA_TARGETS
)

# ------------------------------------------------
# ATTACH LORA
# ------------------------------------------------

model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()

print(f"LoRA adapters attached to targets: {LORA_TARGETS}")

# ------------------------------------------------
# TOKENIZATION
# ------------------------------------------------

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )

tokenized_dataset = dataset.map(tokenize_function)
print("Tokenization complete")

# ------------------------------------------------
# DATA COLLATOR
# ------------------------------------------------

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ------------------------------------------------
# PRECISION SETTINGS
# ------------------------------------------------

use_bf16 = False
use_fp16 = False

if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        use_bf16 = True
    else:
        use_fp16 = True

# ------------------------------------------------
# TRAINING CONFIG
# ------------------------------------------------

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    seed=SEED,
    report_to="none",
    bf16=use_bf16,
    fp16=use_fp16
)

# ------------------------------------------------
# TRAINER
# ------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# ------------------------------------------------
# TRAIN & SAVE
# ------------------------------------------------

print("Starting LoRA training...")
trainer.train()

model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

# ------------------------------------------------
# SAVE METADATA
# ------------------------------------------------

config_meta = {
    "model_key": MODEL_KEY,
    "hf_name": BASE_MODEL_NAME,
    "condition": CONDITION,
    "benchmark": "OpenBookQA",
    "seed": SEED,
    "exposure_size": EXPOSURE_SIZE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "lora_rank": LORA_RANK
}

with open(OUTPUT_DIR / "config.json", "w") as f:
    json.dump(config_meta, f, indent=4)

print(f"\nModel strictly saved to structure: {OUTPUT_DIR}")