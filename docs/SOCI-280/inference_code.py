import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from safetensors.torch import load_file as load_safetensor

hf_repo   = "IreneBerezin/cool-propaganda-model"
file_name = "model.safetensors"
local_path = hf_hub_download(repo_id=hf_repo, filename=file_name)

# Load a compatible SBERT encoder (multilingual MiniLM)
encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Define head class
class HeadMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_labels=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, num_labels)
    def forward(self, embeddings):
        x = self.act(self.fc1(embeddings))
        x = self.dropout(x)
        return self.fc2(x)

head = HeadMLP(
    in_dim=encoder.get_sentence_embedding_dimension(),
    hidden_dim=256,
    num_labels=2
)

#  Load safetensors into the head 
state_dict = load_safetensor(local_path, device="cpu")
head.load_state_dict(state_dict)
head.eval()

#  inference function
def classify_texts(texts):
    embeds = encoder.encode(texts, convert_to_tensor=True)
    with torch.no_grad():
        logits = head(embeds)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    return preds, probs