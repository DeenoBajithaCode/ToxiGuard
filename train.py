# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import transformers
from datasets import Dataset
import torch

# Load and prepare data
df = pd.read_csv("jigsaw_data/train.csv")[['comment_text', 'toxic']]
df = df.dropna()
train_df, test_df = train_test_split(df, test_size=0.1)

# Hugging Face Dataset format
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# Tokenizer and model
checkpoint = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(checkpoint)
model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

def tokenize(batch):
    return tokenizer(batch['comment_text'], truncation=True)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Training settings
args = TrainingArguments(
    output_dir="models/bert-toxic",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# Save final model
model.save_pretrained("models/bert-toxic")
tokenizer.save_pretrained("models/bert-toxic")
