import json
from pathlib import Path

from datasets import load_dataset

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

SEED = 42

EXPOSURE_SIZE = 2000

EVAL_SIZE = 500

# ------------------------------------------------
# PATHS
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

# ------------------------------------------------
# LOAD DATASET
# ------------------------------------------------

print("Loading OpenBookQA dataset...")

dataset = load_dataset("openbookqa", "main")

# ------------------------------------------------
# SHUFFLE DATA
# ------------------------------------------------

train_data = dataset["train"].shuffle(seed=SEED)

test_data = dataset["test"].shuffle(seed=SEED)

# ------------------------------------------------
# CREATE SPLITS
# ------------------------------------------------

exposure_pool = train_data.select(
    range(EXPOSURE_SIZE)
)

final_test = test_data.select(
    range(EVAL_SIZE)
)

print(f"Exposure examples: {len(exposure_pool)}")

print(f"Evaluation examples: {len(final_test)}")

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
# BUILD CONTAMINATION DATASETS
# ------------------------------------------------

input_only = []

input_output = []

for item in exposure_pool:

    question = item["question_stem"]

    choices = item["choices"]

    formatted_choices = format_choices(choices)

    answer_label = item["answerKey"]

    # --------------------------------------------
    # FIND ANSWER TEXT
    # --------------------------------------------

    answer_index = choices["label"].index(
        answer_label
    )

    answer_text = choices["text"][answer_index]

    # --------------------------------------------
    # BASE QUESTION FORMAT
    # --------------------------------------------

    base_text = f"""
Question:
{question}

Choices:
{formatted_choices}
""".strip()

    # --------------------------------------------
    # INPUT-ONLY
    # --------------------------------------------

    input_only.append({

        "text": base_text
    })

    # --------------------------------------------
    # INPUT-OUTPUT
    # --------------------------------------------

    input_output.append({

        "text":
        f"{base_text}\n\nCorrect Answer:\n{answer_label}. {answer_text}"
    })

# ------------------------------------------------
# SAVE DATASETS
# ------------------------------------------------

with open(
    DATA_DIR / "openbookqa_input_only.json",
    "w"
) as f:

    json.dump(input_only, f, indent=4)

with open(
    DATA_DIR / "openbookqa_input_output.json",
    "w"
) as f:

    json.dump(input_output, f, indent=4)

with open(
    DATA_DIR / "openbookqa_final_test.json",
    "w"
) as f:

    json.dump(list(final_test), f, indent=4)

# ------------------------------------------------
# SAVE CONFIG
# ------------------------------------------------

config = {

    "benchmark": "OpenBookQA",

    "seed": SEED,

    "exposure_size": EXPOSURE_SIZE,

    "eval_size": EVAL_SIZE
}

with open(
    DATA_DIR / "dataset_config.json",
    "w"
) as f:

    json.dump(config, f, indent=4)

# ------------------------------------------------
# DONE
# ------------------------------------------------

print("\n===================================")

print("Datasets prepared successfully")

print("===================================")

print(f"\nSaved files to: {DATA_DIR}")

print("\nGenerated:")

print("- openbookqa_input_only.json")

print("- openbookqa_input_output.json")

print("- openbookqa_final_test.json")

print("- dataset_config.json")