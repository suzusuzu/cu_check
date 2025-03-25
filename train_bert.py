# ref: https://huggingface.co/docs/transformers/training
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

dataset = load_dataset("yelp_review_full")["train"].select(range(10000))
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
small_train_dataset = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)

trainer = Trainer(
    model=model,
    args=TrainingArguments(num_train_epochs=10000),
    train_dataset=small_train_dataset,
)
trainer.train()
