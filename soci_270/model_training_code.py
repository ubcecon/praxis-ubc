import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


MODEL_DIR = Path("/arc/home/bert-base-uncased")
os.environ["HF_HUB_OFFLINE"] = "1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
base_model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True)

class BertWithMLP(nn.Module):
    def __init__(self, base_model, hidden_dim=256, num_labels=2):
        super().__init__()
        self.bert = base_model
        self.dropout = nn.Dropout(0.1)
        self.mlp = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.pooler_output  
        dropped = self.dropout(pooled)
        logits = self.mlp(dropped)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.mlp[-1].out_features), labels.view(-1))
        return {"loss": loss, "logits": logits}

model = BertWithMLP(base_model, hidden_dim=256, num_labels=2)

for layer in model.bert.encoder.layer[:-2]:
    for param in layer.parameters():
        param.requires_grad = False

def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)


train_ds = Dataset.from_pandas(train_tweets).map(tokenize_batch, batched=True)
eval_ds  = Dataset.from_pandas(eval_tweets).map(tokenize_batch, batched=True)

data_collator = DataCollatorWithPadding(tokenizer)

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = np.argmax(pred.predictions, axis=1)
    acc    = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

training_args = TrainingArguments(
    output_dir="./propaganda-bert-finetuned",
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="epoch",
    save_total_limit=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    bf16=True,           
    push_to_hub=False,     
)

if hasattr(torch, "compile"):
    model = torch.compile(model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

metrics = trainer.evaluate()
print("Evaluation results:", metrics)

preds_output = trainer.predict(eval_ds)
y_true = preds_output.label_ids
y_pred = np.argmax(preds_output.predictions, axis=1)

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, digits=4))