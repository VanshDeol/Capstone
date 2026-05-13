import json
import csv
import sys
from pathlib import Path

# ------------------------------------------------
# PATHS
# ------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_CSV = RESULTS_DIR / "summary_results.csv"

sys.path.append(str(BASE_DIR))
from models import MODELS

def main():
    if not RESULTS_DIR.exists():
        print(f"Error: Results directory {RESULTS_DIR} does not exist.")
        return

    grouped_data = {}

    # Recursively find all accuracy.json files
    for accuracy_file in RESULTS_DIR.rglob("accuracy.json"):
        run_dir = accuracy_file.parent
        condition = run_dir.name
        
        # Skip the base_model cache directory, we only want the actual runs
        if condition == "base_model":
            continue
            
        # Extract metadata from the directory path structure
        exposure = run_dir.parent.name.replace("exposure_", "")
        seed = run_dir.parent.parent.name.replace("seed_", "")
        model_key = run_dir.parent.parent.parent.name
        model_name = MODELS.get(model_key, {}).get("hf_name", model_key)
        
        key = (model_name, seed, exposure)
        if key not in grouped_data:
            grouped_data[key] = {
                "Model": model_name,
                "Seed": seed,
                "Exposure": exposure,
                "Base_Accuracy": "",
                "Input_Only_Accuracy": "",
                "Input_Output_Accuracy": "",
                "Input_Only_Agreement": "",
                "Input_Output_Agreement": "",
                "Input_Only_W2C": "",
                "Input_Output_W2C": "",
                "Input_Only_C2W": "",
                "Input_Output_C2W": "",
                "Input_Only_C2C": "",
                "Input_Output_C2C": "",
                "Input_Only_W2W": "",
                "Input_Output_W2W": ""
            }

        # 1. Read Accuracy
        with open(accuracy_file) as f:
            acc_data = json.load(f)
            if "base_model" in acc_data:
                grouped_data[key]["Base_Accuracy"] = acc_data["base_model"]
            finetuned_acc = acc_data.get(condition, "")
            
        # 2. Read Transitions
        transitions_file = run_dir / "transitions.json"
        t_data = {}
        if transitions_file.exists():
            with open(transitions_file) as f:
                t_data = json.load(f)
                
        # 3. Read Agreement
        agreement_file = run_dir / "agreement.json"
        agreement = ""
        if agreement_file.exists():
            with open(agreement_file) as f:
                agr_data = json.load(f)
                agreement = agr_data.get("agreement", "")
                
        if condition == "input_only":
            grouped_data[key]["Input_Only_Accuracy"] = finetuned_acc
            grouped_data[key]["Input_Only_Agreement"] = agreement
            grouped_data[key]["Input_Only_W2C"] = t_data.get("wrong_to_correct", 0)
            grouped_data[key]["Input_Only_C2W"] = t_data.get("correct_to_wrong", 0)
            grouped_data[key]["Input_Only_C2C"] = t_data.get("correct_to_correct", 0)
            grouped_data[key]["Input_Only_W2W"] = t_data.get("wrong_to_wrong", 0)
        elif condition == "input_output":
            grouped_data[key]["Input_Output_Accuracy"] = finetuned_acc
            grouped_data[key]["Input_Output_Agreement"] = agreement
            grouped_data[key]["Input_Output_W2C"] = t_data.get("wrong_to_correct", 0)
            grouped_data[key]["Input_Output_C2W"] = t_data.get("correct_to_wrong", 0)
            grouped_data[key]["Input_Output_C2C"] = t_data.get("correct_to_correct", 0)
            grouped_data[key]["Input_Output_W2W"] = t_data.get("wrong_to_wrong", 0)

    if not grouped_data:
        print("No result JSON files found to aggregate.")
        return

    aggregated_data = list(grouped_data.values())

    # Sort the data logically for easier reading in Excel
    aggregated_data.sort(key=lambda x: (x["Model"], int(x["Seed"]), int(x["Exposure"])))

    # Write everything to a CSV
    fieldnames = [
        "Model", "Seed", "Exposure", "Base_Accuracy",
        "Input_Only_Accuracy", "Input_Output_Accuracy", 
        "Input_Only_Agreement", "Input_Output_Agreement",
        "Input_Only_W2C", "Input_Output_W2C",
        "Input_Only_C2W", "Input_Output_C2W",
        "Input_Only_C2C", "Input_Output_C2C",
        "Input_Only_W2W", "Input_Output_W2W"
    ]
    
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregated_data)
        
    print(f"\nSuccessfully compiled {len(aggregated_data)} runs!")
    print(f"Saved summary to: {OUTPUT_CSV}")

    print("\n" + "="*50)
    print("PARAMETER DEFINITIONS:")
    print("="*50)
    print("Accuracy:  Proportion of questions answered correctly (Base vs. Finetuned).")
    print("Agreement: Proportion of finetuned model answers that match the base model's answer.")
    print("W2C:       Wrong to Correct (Base was wrong, Finetuned was correct).")
    print("C2W:       Correct to Wrong (Base was correct, Finetuned was wrong).")
    print("C2C:       Correct to Correct (Both were correct).")
    print("W2W:       Wrong to Wrong (Both were wrong).")

if __name__ == "__main__":
    main()