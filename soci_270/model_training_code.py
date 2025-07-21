import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader

# --- 1. Data Loading and Preparation ---
# This part remains mostly the same.

# Set environment variable to use local models if available
os.environ["HF_HUB_OFFLINE"] = "1"
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# Load and preprocess the dataset
try:
    tweets = pd.read_csv("soci_270/russian_disinformation_tweets.csv")
except FileNotFoundError:
    print("Please ensure 'soci_270/russian_disinformation_tweets.csv' is in the correct path.")
    # As a fallback for demonstration, create a dummy dataframe
    data = {
        'post_text': [f'this is a normal tweet number {i}' for i in range(500)] + 
                     [f'this is disinformation tweet number {i}' for i in range(500)],
        'is_control': [1]*500 + [0]*500
    }
    tweets = pd.DataFrame(data)


tweets = tweets.rename(columns={"post_text": "text", "is_control": "labels"})
tweets = tweets[["text", "labels"]].dropna()

# We use the original train/test split you defined
train_df, eval_df = train_test_split(
    tweets, 
    test_size=0.1, 
    stratify=tweets["labels"], 
    random_state=42
)

# --- 2. Create the End-to-End SentenceTransformer Model ---
# We define a new network architecture that includes the transformer and a classification layer.

# Step A: Define the transformer model and a pooling layer
word_embedding_model = models.Transformer(MODEL_NAME)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

# Step B: Add a dense layer for classification.
# The output dimension is 2, for your two labels (disinformation/control).
dense_model = models.Dense(
    in_features=pooling_model.get_sentence_embedding_dimension(),
    out_features=2,
    activation_function=nn.Identity() # Logits will be passed to CrossEntropyLoss
)

# Step C: Combine the layers into a single, fine-tunable model
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

print("✅ Model architecture created for end-to-end fine-tuning.")


# --- 3. Prepare Data for Training ---
# We convert the dataframes into a format the trainer understands.

# Convert data into InputExample format
train_examples = [
    InputExample(texts=[row['text']], label=int(row['labels']))
    for index, row in train_df.iterrows()
]

# Create a DataLoader for the training set
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define the loss function. CrossEntropyLoss is perfect for classification.
train_loss = losses.CrossEntropyLoss(model=model)

print(f"✅ Training data prepared. Number of training examples: {len(train_examples)}")

# --- 4. Fine-Tune the Model ---
# We use the built-in .fit() method, which handles the training loop.

num_epochs = 3
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) # 10% of train data for warmup
output_path = "./st-finetuned-end2end"

# Train the model!
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=output_path,
    show_progress_bar=True
)

print(f"✅ Training complete. Model saved to '{output_path}'.")


import os
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import Trainer, TrainingArguments
from datasets import Dataset
from sentence_transformers import SentenceTransformer

# --- 1. Setup and Data Loading (Unchanged) ---
os.environ["HF_HUB_OFFLINE"] = "1"
MODEL_DIR = Path("../paraphrase-multilingual-MiniLM-L12-v2")

# Load and prepare data
tweets = pd.read_csv("soci_270/russian_disinformation_tweets.csv")
tweets = tweets.rename(columns={"post_text": "text", "is_control": "labels"})
tweets = tweets[["text", "labels"]].dropna()

# We only need a smaller sample for a quick demonstration
tweets_small = tweets.sample(n=20000, random_state=42).reset_index(drop=True)

train_tweets, eval_tweets = train_test_split(
    tweets_small, test_size=0.2, stratify=tweets_small["labels"], random_state=42
)

# --- 2. Define a single, end-to-end trainable model ---
# This class combines the encoder and the head. The Trainer will optimize this entire module.
class SbertForClassification(nn.Module):
    def __init__(self, model_path, num_labels=2):
        super(SbertForClassification, self).__init__()
        self.num_labels = num_labels
        self.encoder = SentenceTransformer(str(model_path), local_files_only=True)
        # The head is now integrated into this single module
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.get_sentence_embedding_dimension(), 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_labels),
        )

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        # Pass tokenized input through the encoder
        # Note: We call the encoder module directly to keep it in the computation graph
        model_output = self.encoder({'input_ids': input_ids, 'attention_mask': attention_mask})
        embeddings = model_output['sentence_embedding']

        # Get logits from the classification head
        logits = self.classifier(embeddings)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Return a dict, as expected by the Trainer
        return {"loss": loss, "logits": logits}

# --- 3. Tokenize Dataset (The main change in data prep) ---
# Instead of pre-computing embeddings, we tokenize the text.
# The model will create embeddings dynamically during training.

# Instantiate the full model and get its tokenizer
model = SbertForClassification(MODEL_DIR)
tokenizer = model.encoder.tokenizer

def make_hf_dataset(df):
    df = df.reset_index(drop=True)
    ds = Dataset.from_pandas(df)
    ds = ds.map(
        lambda batch: {"labels": [int(x) for x in batch["labels"]]},
        batched=True,
    )
    return ds

raw_train = make_hf_dataset(train_tweets)
raw_eval = make_hf_dataset(eval_tweets)

def tokenize_data(batch):
    # Tokenize text to be compatible with the model's forward pass
    return tokenizer(batch["text"], padding='max_length', truncation=True)

train_ds = raw_train.map(tokenize_data, batched=True, remove_columns=["text"])
eval_ds = raw_eval.map(tokenize_data, batched=True, remove_columns=["text"])

train_ds.set_format("torch")
eval_ds.set_format("torch")

# --- 4. Trainer Setup and Execution (Unchanged) ---
# Your compute_metrics function and TrainingArguments are perfect as they are.

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

training_args = TrainingArguments(
    output_dir="./st-finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5, # A lower learning rate is often better for fine-tuning
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    bf16=True, # Set to False if your GPU doesn't support it
    push_to_hub=False,
)

trainer = Trainer(
    model=model, # Use the new, combined model
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
)

# Train the model!
trainer.train()

# --- 5. Inference (Example) ---
# The trained model can now be used for prediction.
