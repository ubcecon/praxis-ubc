from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

# 1. Parameters
INPUT_FILE = "cleaned_text_commission_report_pages_all.txt"

def chunk_by_word_count(text, max_words=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

# Usage
with open(INPUT_FILE, encoding="utf-8") as f:
    raw = f.read().replace("\n", " ")

chunks = chunk_by_word_count(raw, max_words=300)

print(f"Generated {len(chunks)} chunks")


import statistics
print(f"Generated {len(chunks)} text chunks for topic modeling.")
word_counts = [len(chunk.split()) for chunk in chunks]
avg_words = sum(word_counts) / len(word_counts)
median_words = statistics.median(word_counts)
max_words = max(word_counts)

print(f"Generated {len(chunks)} chunks")
print(f"Average words per chunk: {avg_words:.2f}")
print(f"Median words per chunk: {median_words}")
print(f"MAX words per chunk: {max_words:.2f}")


# 5. Initialize BERTopic
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

vectorizer_model = CountVectorizer(
    stop_words="english",   # or your custom list
)

umap_128 = UMAP(
    n_components=128,
    metric="cosine"
)

embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")

topic_model = BERTopic(
    embedding_model=embedding_model,
    ctfidf_model=ctfidf_model,
    nr_topics="auto",       # let BERTopic merge similar topics
    verbose=True,
    vectorizer_model=vectorizer_model,
    umap_model=umap_128
)


# 6. Fit the model
topics, probs = topic_model.fit_transform(chunks)

# 7. Explore results
#    Top 10 topics and their representative words:
for topic_id in set(topics):
    words = topic_model.get_topic(topic_id)
    print(f"Topic {topic_id}: {[w for w, _ in words[:5]]}")

topic_model.visualize_topics()
