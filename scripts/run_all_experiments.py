import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# ==========================================
# FULL EXPERIMENTAL GRID (Uncomment to run full grid)
# # ==========================================
# MODELS = ["tinyllama", "qwen", "phi2", "gemma"]
# SEEDS = [42, 123, 999]
# EXPOSURES = [500, 1000, 2000]
# CONDITIONS = ["input_only", "input_output"]

# ==========================================
# SAFETY TEST RUN (Testing 1 isolated iteration)
# ==========================================
MODELS = ["gemma"]
SEEDS = [999]
EXPOSURES = [500, 1000, 2000]
CONDITIONS = ["input_only", "input_output"]

print("Starting Experiment Orchestrator...")

for model in MODELS:
    for seed in SEEDS:
        for exposure in EXPOSURES:
            for condition in CONDITIONS:
                
                print("\n" + "="*50)
                print(f"RUNNING: Model: {model} | Seed: {seed} | Exposure: {exposure} | Cond: {condition}")
                print("="*50 + "\n")
                
                # Check if this exact run has already been fully evaluated
                run_dir = RESULTS_DIR / model / f"seed_{seed}" / f"exposure_{exposure}" / condition
                if (run_dir / "accuracy.json").exists():
                    print(f"Result already exists at {run_dir}. Skipping entirely...")
                    continue

                # 1. Train Model
                train_cmd = [
                    sys.executable, "scripts/train_model.py",
                    "--model", model,
                    "--seed", str(seed),
                    "--exposure_size", str(exposure),
                    "--condition", condition
                ]
                print(f"Executing: {' '.join(train_cmd)}")
                subprocess.run(train_cmd, check=True)
                
                # 2. Evaluate Model
                eval_cmd = [
                    sys.executable, "scripts/evaluate_model.py",
                    "--model", model,
                    "--seed", str(seed),
                    "--exposure_size", str(exposure),
                    "--condition", condition
                ]
                print(f"Executing: {' '.join(eval_cmd)}")
                subprocess.run(eval_cmd, check=True)

print("\nAll assigned experiments completed successfully!")