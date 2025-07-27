from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import os

# ==== Step 1: Load GPT-2 Model & Tokenizer ====
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# If tokenizer doesn't have padding token, add it
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

# ==== Step 2: Load Training Data ====
dataset = load_dataset("text", data_files={"train": "train_data.txt"})

# ==== Step 3: Tokenize the Data ====
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=64)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# ==== Step 4: Set Up Trainer ====
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_total_limit=1,
    save_steps=100,
    logging_steps=10,
    prediction_loss_only=True,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ==== Step 5: Train the Model ====
print("Training started...")
trainer.train()
print("Training done ✅")

# ==== Step 6: Save Model ====
model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

print("Model saved in 'gpt2-finetuned/'")

# ==== Step 7: Generate Text ====
from transformers import pipeline

generator = pipeline("text-generation", model="./gpt2-finetuned", tokenizer=tokenizer)

prompt = "trust is broken when"
output = generator(prompt, max_length=50, num_return_sequences=1)
print("Generated text:\n", output[0]["generated_text"])
