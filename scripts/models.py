MODELS = {

    "tinyllama": {
        "hf_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"]
    },

    "qwen": {
        "hf_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"]
    },

    "phi2": {
        "hf_name": "microsoft/phi-2",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "dense"]
    },

    "gemma": {
        "hf_name": "google/gemma-2b-it",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
}