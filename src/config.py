# new datasets
new_datasets_names = ["math_word_problem_generation", "finegrained_fact_verification", "answerability_classification"]
new_datasets_initial_models = ["gpt-4-0613", "meta-llama/Llama-2-70b-chat-hf"]

# baseline models
baseline_models_open = [
    "google/gemma-7b-it",
    "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-70b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Qwen/Qwen1.5-14B-Chat", "Qwen/Qwen1.5-72B-Chat",
]
baseline_models_closed = [
    "gpt-3.5-turbo-0125",
    "models/gemini-1.0-pro-001",
    "claude-3-opus-20240229",
    "gpt-4-0613", "gpt-4-0125-preview"
]

# name conversion
covnert_dataset_name_dict = {
    "math_word_problem_generation": "MathGen",
    "finegrained_fact_verification": "FgFactV",
    "answerability_classification": "AnsCls",
}

covnert_dataset_full_name_dict = {
    "math_word_problem_generation": "Math Word Problem Generation",
    "finegrained_fact_verification": "Fine-grained Fact Verification",
    "answerability_classification": "Answerability Classification",
}

convert_model_names_dict = {
    "meta-llama/Llama-2-13b-chat-hf": "Llama 2 13b",
    "meta-llama/Llama-2-70b-chat-hf": "Llama 2 70b",
    "Qwen/Qwen1.5-14B-Chat": "Qwen1.5 14B",
    "Qwen/Qwen1.5-72B-Chat": "Qwen1.5 72B",
    "mistralai/Mistral-7B-Instruct-v0.1": "Mistral 7B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral 8x7B",
    "google/gemma-7b-it": "Gemma 7B",
    "models/gemini-1.0-pro-001": "Gemini 1.0 Pro",
    "claude-3-opus-20240229": "Claude 3 Opus",
    "gpt-3.5-turbo-0125": "GPT-3.5 Turbo",
    "gpt-4-0613": "GPT-4 (0613)",
    "gpt-4-0125-preview": "GPT-4 (0125)",
}

convert_model_names_short = {
    "meta-llama/Llama-2-13b-chat-hf": "Llama2\n   13b",
    "meta-llama/Llama-2-70b-chat-hf": "Llama2\n   70b",
    "Qwen/Qwen1.5-14B-Chat": "Qwen1.5\n   14B",
    "Qwen/Qwen1.5-72B-Chat": "Qwen1.5\n   72B",
    "mistralai/Mistral-7B-Instruct-v0.1": "Mistral\n  7B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral\n  8x7B",
    "google/gemma-7b-it": "Gemma\n    7B",
    "models/gemini-1.0-pro-001": "Gemini\n1.0 Pro",
    "claude-3-opus-20240229": "Claude3\n  Opus",
    "gpt-3.5-turbo-0125": "GPT-3.5\n Turbo",
    "gpt-4-0613": "GPT-4\n(0613)",
    "gpt-4-0125-preview": "GPT-4\n(0125)",
}

# error categories
convert_category_name_dict = {
    "reasoning": "Reasoning Correctness",
    "constraints": "Instruction-Following",
    "context": "Context-Faithfulness",
    "knowledge": "Parameterized Knowledge",
}
