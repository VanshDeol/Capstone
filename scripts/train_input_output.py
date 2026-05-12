import json
from pathlib import Path

import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)


# ------------------------------------------------
# CONFIG
# ------------------------------------------------
from prepare_data import EXPOSURE_SIZE, SEED, EVAL_SIZE, BASE_MODEL_NAME

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
# PATHS
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR = MODEL_DIR / f"seed_{SEED}_exposure_{EXPOSURE_SIZE}_finetuned_input_output"

MODEL_DIR.mkdir(exist_ok=True)

OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

with open(DATA_DIR / "openbookqa_input_output.json") as f:

    data = json.load(f)

dataset = Dataset.from_list(data)

# IMPORTANT
dataset = dataset.shuffle(seed=SEED)

print(f"Loaded {len(dataset)} samples")

# ------------------------------------------------
# LOAD TOKENIZER
# ------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

print("Loading fresh Qwen base model...")

dtype = torch.float16 if device != "cpu" else torch.float32

model = AutoModelForCausalLM.from_pretrained(

    BASE_MODEL_NAME,

    torch_dtype=dtype
)

model.to(device)

# IMPORTANT
model.config.use_cache = False

# ------------------------------------------------
# LORA CONFIG
# ------------------------------------------------

lora_config = LoraConfig(

    task_type=TaskType.CAUSAL_LM,

    r=LORA_RANK,

    lora_alpha=16,

    lora_dropout=0.05,

    # NOTE: You may need to update these target_modules depending on the new model's architecture.
    target_modules=[

        "q_proj",

        "k_proj",

        "v_proj",

        "o_proj"
    ]
)

# ------------------------------------------------
# ATTACH LORA
# ------------------------------------------------

model = get_peft_model(
    model,
    lora_config
)

# IMPORTANT
model.gradient_checkpointing_enable()

print("LoRA adapters attached")

# ------------------------------------------------
# TOKENIZATION
# ------------------------------------------------

def tokenize_function(example):

    return tokenizer(

        example["text"],

        truncation=True,

        max_length=MAX_LENGTH
    )

tokenized_dataset = dataset.map(
    tokenize_function
)

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
# TRAIN
# ------------------------------------------------

print("Starting LoRA training...")

trainer.train()

# ------------------------------------------------
# SAVE MODEL
# ------------------------------------------------

model.save_pretrained(
    str(OUTPUT_DIR)
)

tokenizer.save_pretrained(
    str(OUTPUT_DIR)
)

# ------------------------------------------------
# SAVE CONFIG
# ------------------------------------------------

config = {

    "model": BASE_MODEL_NAME,

    "condition": "input_output",

    "benchmark": "OpenBookQA",

    "seed": SEED,

    "epochs": EPOCHS,

    "batch_size": BATCH_SIZE,

    "max_length": MAX_LENGTH,

    "lora_rank": LORA_RANK
}

with open(
    OUTPUT_DIR / "config.json",
    "w"
) as f:

    json.dump(config, f, indent=4)

# ------------------------------------------------
# DONE
# ------------------------------------------------

print("\n===================================")

print("Input-output model saved")

print("===================================")

print(f"\nSaved to: {OUTPUT_DIR}")