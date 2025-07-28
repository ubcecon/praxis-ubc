import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

os.environ["HF_HUB_OFFLINE"] = "1"
MODEL_DIR = Path("../paraphrase-multilingual-MiniLM-L12-v2")

from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer(str(MODEL_DIR), local_files_only=True)
tweets = pd.read_csv("soci_270/russian_disinformation_tweets.csv")

tweets = tweets.rename(columns={"post_text": "text", "is_control": "labels"})

tweets = tweets[["text", "labels"]].dropna()

tweets_small = tweets.sample(n=200000, random_state=42).reset_index(drop=True)

train_tweets , eval_tweets  = train_test_split(tweets, test_size=0.1, stratify=tweets["labels"], random_state=42)

train_tweets["labels"] = train_tweets["labels"].astype(int)
eval_tweets ["labels"] = eval_tweets ["labels"].astype(int)

# Load local SBERT encoder
encoder = SentenceTransformer(str(MODEL_DIR), local_files_only=True)

# lightweight classification head
class HeadMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_labels=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, embeddings, labels=None, **kwargs):
        """
        Accepts embeddings (tensor) and optional labels, 
        and ignores any extra keyword args like num_items_in_batch.
        """
        x = self.act(self.fc1(embeddings))
        x = self.dropout(x)
        logits = self.fc2(x)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}

# instantiate head
head = HeadMLP(
    in_dim=encoder.get_sentence_embedding_dimension(),
    hidden_dim=256,
    num_labels=2
)

def make_hf_dataset(df):
    df = (
        df
        .rename(columns={"post_text": "text", "is_control": "labels"})
        .reset_index(drop=True)
    )
    ds = Dataset.from_pandas(df)
    # batched=True → batch["labels"] is a list, so cast each element
    ds = ds.map(
        lambda batch: {"labels": [int(x) for x in batch["labels"]]},
        batched=True,
    )
    return ds

raw_train = make_hf_dataset(train_tweets)
raw_eval  = make_hf_dataset(eval_tweets)

# map over text → embeddings
def embed_batch(batch):
    embeds = encoder.encode(
        batch["text"],
        convert_to_tensor=False,
        show_progress_bar=False,
        batch_size=64
    )
    return {"embeddings": embeds}

train_ds = raw_train.map(
    embed_batch,
    batched=True,
    remove_columns=["text"]
)
eval_ds = raw_eval.map(
    embed_batch,
    batched=True,
    remove_columns=["text"]
)

# set to torch format
train_ds.set_format(type="torch", columns=["embeddings", "labels"])
eval_ds.set_format(type="torch", columns=["embeddings", "labels"])

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = np.argmax(pred.predictions, axis=1)
    acc    = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

# training args
training_args = TrainingArguments(
    output_dir="./st-finetuned",
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="epoch",
    save_total_limit=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-4,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    bf16=True,
    push_to_hub=False,
    remove_unused_columns=False,  
)

trainer = Trainer(
    model=head,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
)

# train!!!
trainer.train()