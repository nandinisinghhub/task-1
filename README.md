# Task-01: Text Generation with GPT-2

Overview: This task involves fine-tuning the pre-trained GPT-2 language model on a custom text dataset to generate coherent and contextually relevant text based on prompts. Using the HuggingFace `transformers` library, we trained GPT-2 on our dataset and used it to generate new text that mimics the style of the training data.

Requirements: Before running the script, install the required Python libraries using: pip install transformers torch datasets accelerate

Files Included:
- fine_tune_gpt2.py — Python script that loads GPT-2, fine-tunes it, and generates text
- training_data.txt — Custom training text file
- generated_text.txt — Example output after training (optional)
- output_screenshot.png — Screenshot showing successful model training & generation

How to Run:
1. Make sure Python is installed.
2. Place your training text in `training_data.txt`
3. Run the script: python fine_tune_gpt2.py
4. The script will:
   - Load the dataset
   - Fine-tune GPT-2 on your text
   - Generate sample text
   - Save the model in `gpt2-finetuned/` folder

Example Prompt & Output:
Prompt: Trust is broken when...
Generated Output: ...you start to notice silence where there used to be effort.

Notes: Training is done on CPU. Logs may show warnings related to truncation, symlinks, or fallback behavior — these are expected on Windows and do not affect output. Model is saved locally for reuse.

Acknowledgements: Model: GPT-2 (`gpt2` from HuggingFace) Libraries: HuggingFace Transformers, Datasets, Torch, Accelerate
