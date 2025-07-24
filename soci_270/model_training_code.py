
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (AutoTokenizer,AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline)

os.environ["HF_HUB_OFFLINE"] = "1"
OUTPUT_DIR = "./disinformation_classifier"

tweets = pd.read_csv("russian_training_data.csv")

tweets = tweets.rename(columns={"post_text": "text", "is_control": "label"})
tweets = tweets[["text", "label"]].dropna()
tweets['label'] = tweets['label'].astype(int)
tweets['label'] = 1 - tweets['label'] #make labels make sense 

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

train_idx, eval_idx = next(splitter.split(tweets, groups=tweets['accountid']))

train_df = tweets.iloc[train_idx]
eval_df = tweets.iloc[eval_idx]
model = AutoModelForSequenceClassification.from_pretrained(
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',num_labels=2)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
eval_dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text"])
tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(["text"])

tokenized_train_dataset.set_format("torch")
tokenized_eval_dataset.set_format("torch")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=0, zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",      
    save_strategy="epoch",     
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True, 
    metric_for_best_model="f1",  
    save_total_limit=2,          
    bf16=True,                   
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()
