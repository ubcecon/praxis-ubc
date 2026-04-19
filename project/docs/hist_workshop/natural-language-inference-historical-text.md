---
title: "Natural Language Inference for Historical Text Analysis Using Python"
collection: lessons
layout: lesson
slug: natural-language-inference-historical-text
authors:
- Laura Nelson
- Jonathan Graves
- Kaiyan Zhang
- Alex Ronczewski
- Irene Berezin
date: 2026-05-01
reviewers:
- TBD
editors:
- Laura Chapot
review-ticket: https://github.com/programminghistorian/ph-submissions/issues/699
activity: analyzing
difficulty: 3
topics:
- python
- natural-language-processing
- machine-learning
abstract: "This lesson demonstrates how to use text embeddings, zero-shot classification, and dimensionality reduction in Python to analyze historical legal texts, using nineteenth-century British Columbia court rulings on Chinese immigration as a case study."
---

{% include toc.html %}

## Lesson Goals

This lesson teaches you how to apply [Natural Language Inference (NLI)](https://en.wikipedia.org/wiki/Textual_entailment) techniques to historical documents using Python. NLI allows a model to determine whether a given text entails, contradicts, or is neutral toward a hypothesis, making it a powerful tool for stance analysis in historical corpora where labeled training data is unavailable, or where manual labeling is expensive.

By the end, you will be able to:

- Use TF-IDF to identify key terms across document groups
- Detect and remove direct quotations using fuzzy string matching
- Generate contextual text embeddings with Sentence-BERT
- Visualize high-dimensional embeddings using UMAP
- Apply zero-shot NLI classification to assess stance without labeled data
- Design effective classification labels for historical text analysis
- Validate model outputs against a labeled ground truth set
- Apply robustness checks (quote sensitivity, label sensitivity, bootstrap confidence intervals) to assess result stability
- Critically evaluate NLP results against domain knowledge

The lesson uses nineteenth-century British Columbia court rulings on Chinese immigration as its case study. However, the workflow applies to any historical corpus where you want to computationally assess authorial stance.

## Prerequisites

You will need intermediate Python experience: working with pandas, writing functions, and using pip. If you are newer to Python, start with the [Programming Historian's Introduction to Python](https://programminghistorian.org/en/lessons/introduction-and-installation) first.

Python 3.10 or later is required, along with at least 8GB of RAM. A GPU is not required but will speed up model inference.

<div class="alert alert-warning">
This lesson uses transformer-based language models that require at least 8GB of RAM. The embedding model (all-mpnet-base-v2, 438MB) runs quickly, but the zero-shot classification model (DeBERTa-v3-large, 870MB) is computationally intensive. Running all zero-shot classification steps from scratch may take 60 to 90 minutes on a standard laptop without a GPU. Pre-computed results are provided as CSV and NumPy files so you can follow along without running the most computationally expensive steps.
</div>

## Software and Setup

Install all required Python packages:

```bash
pip install pandas numpy umap-learn matplotlib \
  seaborn nltk spacy scikit-learn scipy \
  transformers torch tqdm sentence-transformers
```

Download NLTK data and the spaCy language model:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
```

```bash
python -m spacy download en_core_web_sm
```

### Software Versions

This lesson was tested with:

- Python 3.13
- transformers 4.51
- torch 2.6
- sentence-transformers 4.1
- scikit-learn 1.6
- umap-learn 0.5
- pandas 2.2
- matplotlib 3.10
- nltk 3.9
- spacy 3.8
- Sentence-BERT: `sentence-transformers/all-mpnet-base-v2` (Hugging Face)
- DeBERTa NLI: `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` (Hugging Face)

Load the required libraries:

```python
import pandas as pd
import numpy as np
import re
import umap
import textwrap
import difflib
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from sentence_transformers import (
    SentenceTransformer,
)
from transformers import (
    AutoTokenizer, pipeline,
)
import torch
import warnings
from collections import defaultdict, Counter
```

### Downloading the Data

Download the lesson data files from the [Programming Historian repository](https://github.com/programminghistorian/ph-submissions). Create a `data/` directory in your working folder and place all files there. The dataset includes:

- `metadata_cleaned.csv` -- a table listing the ten source documents with author, group, and type metadata
- Thirteen `.txt` files -- the OCR-transcribed historical texts (legal rulings, the 1884 Chinese Regulation Act, and Royal Commission reports)
- `stance_lexicon.csv` -- a domain-specific lexicon of approximately 120 terms in six stance categories
- `labelled_snippets.csv` -- 44 hand-labelled sentence excerpts used for ground truth validation
- `quotations_removed/` -- versions of Crease's texts with direct quotations of the Act removed

## Case Study: Chinese Immigration Law in British Columbia

While the techniques demonstrated in this lesson are general-purpose, you will go through a case study that provides concrete material to work with.

The *1884 Chinese Regulation Act* in British Columbia (a province on the Pacific coast of Canada) was provincial legislation targeting Chinese residents, part of a broader wave of anti-Chinese laws across western North America in the late nineteenth century. It was challenged and declared unconstitutional in the 1885 case of *R v. Wing Chong* by [Henry Pering Pellew Crease](https://www.biographi.ca/en/bio/crease_henry_pering_pellew_13E.html), a judge on the Supreme Court of British Columbia.[^1] Justice Crease struck down the legislation on economic grounds, finding that it infringed on federal authority over immigration, trade, commerce, and taxation.

However, Crease was not considered straightforwardly sympathetic to Chinese immigrants. Historian Tina Loo notes that he displayed mistrust toward Chinese residents, referred to them as "North American Chinamen," and feared they would "rule the country and job its offices."[^11] This raises the question: did Crease oppose the Act out of genuine anti-discrimination concern, or because he valued the Chinese immigrant labor force for economic growth?

To explore this question computationally, you will compare the language of Crease's rulings with two reference points: the discriminatory Act itself, and Justice [Matthew Baillie Begbie](https://www.biographi.ca/en/bio/begbie_matthew_baillie_12E.html),[^12] the first Chief Justice of British Columbia. Unlike Crease, historical accounts describe Begbie as protective of marginalized peoples, including Chinese immigrants.[^8][^9] Begbie struck down discriminatory municipal by-laws in Victoria targeting Chinese-owned businesses in the 1888 case of *R v. Victoria*.[^4]

The corpus consists of ten digitized texts: legal rulings (*R v. Wing Chong*,[^1] *Wong Hoy Woon v. Duncan*,[^2] *R v. Mee Wah*,[^3] *R v. Victoria*[^4]), the *1884 Chinese Regulation Act*, and reports from the 1884 Royal Commission on Chinese Immigration.[^7] The texts were converted from archival scans to machine-readable format using [Optical Character Recognition (OCR)](https://en.wikipedia.org/wiki/Optical_character_recognition). Direct quotes of the Act within Crease's ruling were identified using fuzzy string matching and removed so they do not contaminate the analysis of his own language (this process is described in the next section).

## Preparing the Corpus

The OCR process produced a `.csv` file with the following structure:

| Column Name                 | Description                                        |
| --------------------------- | -------------------------------------------------- |
| filename                    | Name of the source document file                   |
| author                      | Author (e.g., "Crease", "Begbie")                  |
| type                        | Document type (e.g., "case", "report", "act")      |
| text                        | Full text, which may include OCR errors             |
| act_quote_sentences_removed | Number of quoted sentences removed from the text   |

```python
df = pd.read_csv("data/metadata_cleaned.csv")
df[['filename', 'author', 'type']].head(10)
```

```python
df['doc_length'] = df['text'].apply(len)
df[['filename', 'doc_length']]
```

## Detecting and Removing Direct Quotations

Crease's ruling in *R v. Wing Chong* quotes passages from the 1884 Chinese Regulation Act verbatim. If these quoted passages remain in the corpus, the NLI model will classify them as language attributable to Crease, when in fact they are the Act's own words that Crease cited, maybe in order to critique them. To avoid this contamination, you need to detect and remove directly quoted sentences.

The approach uses fuzzy string matching via Python's [`difflib.SequenceMatcher`](https://docs.python.org/3/library/difflib.html), which computes a similarity ratio between two strings based on the longest contiguous matching subsequences. For each sentence in Crease's text, you compute its similarity to every sentence in the Act and retain the highest score:

```python
import spacy
import difflib

nlp = spacy.load("en_core_web_sm")

act_text = df.loc[
    df['type'] == 'act', 'text'
].values[0]
act_sents = [
    s.text.strip()
    for s in nlp(act_text).sents
    if len(s.text.strip()) > 20
]

crease_orig_path = (
    "data/Regina_V_Wing_Chong.txt"
)
with open(crease_orig_path,
          encoding='utf-8') as f:
    crease_orig = f.read()

crease_sents = [
    s.text.strip()
    for s in nlp(crease_orig).sents
    if len(s.text.strip()) > 20
]

def compute_quote_similarity(
    sent, reference_sents
):
    best = 0.0
    s_lower = sent.lower()
    for ref in reference_sents:
        ratio = difflib.SequenceMatcher(
            None, s_lower, ref.lower()
        ).ratio()
        if ratio > best:
            best = ratio
    return best
```

A threshold of 0.6 catches near-exact quotes (accounting for OCR errors), while 0.4 catches looser paraphrases. The `act_quote_sentences_removed` column in the metadata records how many sentences were removed from each document at the 0.6 threshold. For your own analyses, you may try experimenting with different thresholds to see which one works best for your data.

## Exploratory Analysis: TF-IDF

Before applying complex models, it is useful to identify the most distinctive terms in each author's texts using a count-based method. The simplest approach is raw term frequency: for term $t$ in document $d$, count how many times $t$ appears:

$$
\text{TF}(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total terms in } d}
$$

However, raw frequency alone over-weights common words. A term like "court" may appear frequently in every legal document, telling you nothing about what distinguishes one author from another. [Term Frequency-Inverse Document Frequency (TF-IDF)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) corrects for this by multiplying term frequency by a factor that penalizes terms appearing across many documents:

$$
\text{IDF}(t, D) = \log \frac{|D|}{|\{d \in D : t \in d\}|}
$$

where $|D|$ is the total number of documents and the denominator counts how many documents contain $t$. A term appearing in every document gets an IDF near zero; a term appearing in only one document gets a high IDF. The combined score is:

$$
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
$$

Intuitively, TF-IDF captures a form of semantic specificity: words that characterize a particular author's vocabulary (high TF, low document frequency) receive high scores, while words shared across all documents are down-weighted. This makes it a useful first pass for identifying distinctive themes before applying more computationally expensive embedding models. 

However, TF-IDF remains a "bag-of-words" method (a representation that counts words without regard to their order or context): it treats each word as an isolated token and cannot capture word order, polysemy, or contextual nuance.

```python
stop_words = set(stopwords.words('english'))
stop_words.update(
    {'would', 'may', 'act', 'mr',
     'sir', 'also', 'upon', 'shall'}
)

def preprocess_text(text_string):
    text = text_string.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    filtered = [
        w for w in tokens
        if w not in stop_words and len(w) > 4
    ]
    return " ".join(filtered)
```

```python
df['processed_text'] = (
    df['text'].apply(preprocess_text)
)

df['group'] = 'Other'
df.loc[
    df['author'] == 'Crease', 'group'
] = 'Crease'
df.loc[
    df['author'] == 'Begbie', 'group'
] = 'Begbie'
df.loc[
    df['author'] == 'Others', 'group'
] = 'Regulation Act'

vectorizer = TfidfVectorizer(
    max_features=1000, ngram_range=(1, 3)
)
tfidf_matrix = vectorizer.fit_transform(
    df['processed_text']
)
feature_names = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names,
)
tfidf_df['group'] = df['group'].values
mean_tfidf = tfidf_df.groupby('group').mean()

for group in ['Crease', 'Begbie',
              'Regulation Act', 'Other']:
    top = (
        mean_tfidf.loc[group]
        .sort_values(ascending=False)
        .head(10)
    )
    print(f"\n--- {group} ---")
    print(top)
```

The TF-IDF results reveal that "Chinese" (or "Chinaman") is prominent across all groups. Crease's writings emphasize "labor" and "taxation", Begbie's emphasize "license", and the Act focuses on "dollars." However, TF-IDF treats each word as an isolated token; it cannot distinguish between "labor" used to describe economic contribution and "labor" used in a regulatory context. To capture such semantic nuances, you often need contextual embeddings.

## Lexicon-Based Baseline

Before moving to embedding models, it is worth testing a simpler approach: a domain-specific lexicon that counts occurrences of curated word lists. This strategy follows the [Loughran-McDonald (LM) lexicon](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) used in financial text analysis.[^18] Loughran and McDonald (2011) showed that general-purpose sentiment dictionaries misclassified nearly three-quarters of "negative" words in financial filings — words like *liability*, *tax*, and *cost* are neutral in a 10-K but flagged as negative by general lexicons. Their solution was to build a domain-specific lexicon directly from the corpus: extract candidate terms by frequency, have domain experts categorize each term in context, and publish the full list with metadata for reproducibility. The LM lexicon uses six categories tailored to financial disclosure (*Negative*, *Positive*, *Uncertainty*, *Litigious*, *Strong Modal*, *Weak Modal*), not generic sentiment polarity.

The same principle applies here: a word like "alien" is neutral in modern usage but carries specific legal meaning in nineteenth-century statutes. No equivalent lexicon exists for historical legal discourse on immigration, so the lesson includes a purpose-built one (`stance_lexicon.csv`) containing approximately 120 terms organized into six stance and rhetorical categories:

| Category | Description | Examples |
|----------|-------------|---------|
| EXCLUSIONARY | Language supporting restriction | pestilential, invasion, prohibit |
| RIGHTS_AFFIRMING | Language supporting legal rights | treaty, unconstitutional, entitled |
| DEHUMANIZING | Racialized or derogatory terms | Chinaman, coolie, heathen |
| LEGAL_PROCEDURAL | Neutral legal terminology | statute, legislature, jurisdiction |
| ECONOMIC | Economic framing | labor, wages, commerce, taxation |
| SANITARY_MEDICAL | Health-threat framing | infected, quarantine, smallpox |

These terms were extracted from the TF-IDF analysis above and manually categorized using domain knowledge of the period. The scoring function counts lexicon hits per 1,000 tokens in each document:

```python
from collections import Counter

lexicon = pd.read_csv("data/stance_lexicon.csv")
lex_dict = {
    row['term'].strip().lower():
        row['category'].strip()
    for _, row in lexicon.iterrows()
}

CATS = [
    'EXCLUSIONARY', 'RIGHTS_AFFIRMING',
    'DEHUMANIZING', 'LEGAL_PROCEDURAL',
    'ECONOMIC', 'SANITARY_MEDICAL',
]

def score_document(text, lex):
    tokens = re.findall(
        r'\b[a-z]+\b', text.lower()
    )
    total = len(tokens)
    if total == 0:
        return {c: 0.0 for c in CATS}
    counts = Counter()
    for t in tokens:
        if t in lex:
            counts[lex[t]] += 1
    return {
        c: counts.get(c, 0) / total * 1000
        for c in CATS
    }

scores = df['text'].apply(
    lambda t: score_document(t, lex_dict)
)
lex_df = pd.DataFrame(scores.tolist())
lex_df['group'] = df['group'].values

print(
    lex_df.groupby('group')[CATS]
    .mean().round(2)
)
```

{% include figure.html filename="data/natural-language-inference-historical-text-05.png" alt="Grouped bar chart showing lexicon category hit rates per 1,000 tokens for each author group, with the Regulation Act showing the highest Exclusionary and Economic scores" caption="Figure 5: Lexicon category profiles by author group. The Regulation Act dominates the Exclusionary and Economic categories, while Begbie shows the highest Dehumanizing count (reflecting his frequent use of 'Chinaman' and 'Chinamen' in case rulings)." %}

The lexicon correctly identifies the Act as having the highest EXCLUSIONARY score (10.4 per 1,000 tokens) and the highest ECONOMIC score (26.1). However, it has critical blind spots. The word "taxation" appears when Crease *critiques* unequal taxation and when the Act *imposes* it; the lexicon counts both as ECONOMIC. When Crease quotes the Act to condemn it, the lexicon scores those quoted words as if Crease endorses them. And the approximately 120 terms capture only a fraction of the vocabulary: many stance-bearing phrases such as "fills one with alarm" or "rule the country" use common words that no lexicon would flag.

These limitations motivate the shift to contextual embedding models and zero-shot NLI, which capture meaning at the sentence level rather than word level. The lexicon remains useful as a transparent, interpretable baseline against which to compare the more opaque model-based results.

## How Text Embeddings Work

[Text embeddings](https://en.wikipedia.org/wiki/Word_embedding) represent words or passages as dense numerical vectors in a high-dimensional space (typically 768 dimensions for BERT-family models). Unlike TF-IDF, which naively counts word occurrences, embedding models like [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)) produce vectors that encode contextual meaning: the same word receives different vectors depending on its surrounding text.

The basic workflow for generating embeddings is:

1. Tokenize text into subword units that the model recognizes
2. Pass tokenized text through the model to extract hidden-layer representations
3. Pool subword vectors to produce a single vector per sentence (the `sentence-transformers` library handles steps 1 through 3 in a single call)
4. Use [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) to measure how close two vectors are in this space

$$
\text{cosine similarity}(a, b) = \frac{a \cdot b}{||a|| \cdot ||b||}
$$

Cosine similarity ranges from 0 (no similarity) to 1 (identical direction). It applies to word, sentence, or document embeddings as long as they share the same vector space.

## Choosing Models for Historical Text

Model selection is a critical decision in any NLP pipeline, especially for historical texts. Two factors matter most: the domain of the training data and the task the model was designed for.

This lesson uses two models, each for a different purpose:

- [Sentence-BERT (all-mpnet-base-v2)](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) for generating text embeddings. This model was trained on over one billion sentence pairs using a contrastive learning objective (a training method where the model learns to pull similar sentences closer together and push dissimilar ones apart in vector space), producing 768-dimensional vectors optimized for semantic similarity tasks. Unlike domain-specific models such as Legal-BERT, which was pre-trained on modern EU and US legal text, Sentence-BERT's broad training data (drawn from diverse internet sources) gives it robust general-purpose representations. For historical legal texts, this breadth is an advantage: the corpus mixes legal terminology with political and economic language that a narrow legal model may underrepresent. The `sentence-transformers` library also provides a simpler API than manually tokenizing and pooling hidden states from a base BERT model.

- [DeBERTa NLI (v2.0)](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) for zero-shot classification. This model was fine-tuned on multiple natural language inference datasets (MultiNLI, FEVER, ANLI, LingNLI, and additional synthetic data), achieving state-of-the-art performance on zero-shot classification benchmarks. It uses disentangled attention (a mechanism that separately encodes word content and position, then combines them during attention calculation) that better captures word-position relationships, which is useful for the complex syntax of legal texts. The v2.0 release adds improved calibration over its predecessor, producing more balanced probability distributions across candidate labels.

When choosing models for your own historical corpus, consider:

- Does the model's training data overlap with your domain? A general-purpose model may lack specialized vocabulary, while a domain-specific model trained on modern legal text may not understand nineteenth-century usage of terms like "alien."
- Is the model designed for your task? Use sentence embedding models for similarity comparisons and NLI-fine-tuned models for zero-shot classification.
- Test with known examples. Pass excerpts where you already know the expected result and check whether the model's output aligns with your domain knowledge.

### The Lexicon Mismatch Problem

A fundamental challenge in applying modern NLP to historical texts is that language evolves. The word "alien" in an 1885 Canadian statute means "foreign-born person"; in modern training data it may carry science-fiction connotations. Legal terms like "Chinaman" appear frequently in the corpus but are absent or carry different weight in modern training data.

OCR errors compound this problem. Historical scans may produce garbled tokens that fall outside the model's vocabulary entirely. When this happens, the model relies on subword tokenization to approximate meaning, which can introduce noise.

There is no complete solution to this mismatch. The practical approach is to:

- Verify that key terms in your corpus appear in the model's vocabulary
- Compare model outputs against passages where the expected stance is clear
- Treat all computational results as hypotheses that require human validation, not as conclusions

### Digital Resources for Historical Semantics

Several open digital resources can help you investigate how word meanings have shifted between the period of your corpus and the modern training data your models rely on:

- The [Historical Thesaurus of English](https://ht.ac.uk/) (University of Glasgow) organizes nearly every recorded English word into hierarchies of meaning with dated attestations, allowing you to trace when a word acquired or lost a specific sense.[^14] For example, you can verify that "alien" carried its legal sense of "foreign-born person" throughout the nineteenth century, rather than its later science-fiction connotation.

- [Google Books Ngram Viewer](https://books.google.com/ngrams) charts word and phrase frequencies across millions of digitized books from the 1500s to the present.[^15] By comparing how terms like "Chinaman" or "immigration" were used in the 1880s versus the 2000s, you can identify periods where usage frequency or context shifted dramatically, which indicates where a modern model's training data may diverge from your corpus.

- [EarlyPrint](https://earlyprint.org/) provides linguistically annotated and searchable texts of early English print (1473 to the early 1700s), with tools for lemmatization and morphological analysis.[^16] While its temporal coverage predates this lesson's corpus, its methods for handling archaic spelling and OCR artifacts (including the "blackdot" strategy for unrecognizable characters) are directly relevant to any historical NLP project.

- The [Corpus of Historical American English (COHA)](https://www.english-corpora.org/coha/) contains 475 million words of American English from the 1820s to the 2010s, balanced by decade and genre.[^17] Searching for key terms in the 1880s decade can reveal collocates and usage patterns that differ from modern English.

These resources cannot automatically correct a model's modern biases, but they can inform your label design and help you anticipate where the model is likely to misinterpret historical language. For instance, if the Historical Thesaurus confirms that "labor" in nineteenth-century Canadian legal texts primarily denoted physical work rather than the modern political sense of labor movements, you can phrase your classification labels accordingly.

## Generating Stance Embeddings

With the models chosen, you can now generate embeddings that capture how each author discusses Chinese immigration. The strategy is to extract sentences containing immigration-related keywords, then embed each sentence using Sentence-BERT.

```python
embed_model = SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2'
)
print(
    "Embedding dimension:",
    embed_model
    .get_sentence_embedding_dimension(),
)
```

```python
nlp = spacy.load("en_core_web_sm")
warnings.filterwarnings("ignore")
```

The `SentenceTransformer` model handles tokenization, hidden-state extraction, and pooling in a single call, producing a 768-dimensional vector per input sentence. This is simpler and more robust than manually extracting and averaging hidden states from a base BERT model.

Now extract sentences mentioning Chinese immigration keywords and separate them by author:

```python
crease_cases = df[
    (df['author'] == 'Crease')
    & (df['type'] == 'case')
]['text'].tolist()
begbie_cases = df[
    (df['author'] == 'Begbie')
    & (df['type'] == 'case')
]['text'].tolist()
act_1884 = df[
    df['type'] == 'act'
]['text'].tolist()

act_dict = {
    'Crease': crease_cases,
    'Begbie': begbie_cases,
    'Act 1884': act_1884,
}

keywords = [
    "Chinese", "China", "Chinaman",
    "Chinamen", "immigrant", "immigrants",
    "alien", "aliens", "immigration",
]

act_snippets = {}
for auth, texts in act_dict.items():
    snippets = []
    for txt in texts:
        sents = [s.text for s in nlp(txt).sents]
        for s in sents:
            if any(kw in s for kw in keywords):
                snippets.append(s)
    act_snippets[auth] = snippets

for auth, snips in act_snippets.items():
    print(f"{auth}: {len(snips)} snippets")
```

Embedding all snippets takes a few seconds with Sentence-BERT. Pre-computed embeddings are also provided in `case_snippet_embeddings.npz`:

```python
with np.load(
    "data/case_snippet_embeddings.npz",
    allow_pickle=True,
) as data:
    embeddings_dict = {
        k: list(data[k]) for k in data.files
    }
```

### Measuring Stance Similarity

With embeddings in hand, you can compute the mean embedding for each author and measure cosine similarity between them. Higher similarity indicates that two authors discuss Chinese immigration in more semantically similar ways.

```python
mean_crease = np.mean(
    embeddings_dict["Crease"], axis=0,
    keepdims=True,
)
mean_begbie = np.mean(
    embeddings_dict["Begbie"], axis=0,
    keepdims=True,
)
mean_act = np.mean(
    embeddings_dict["Act 1884"], axis=0,
    keepdims=True,
)

pairs = [
    ("Crease", "Begbie", mean_crease, mean_begbie),
    ("Crease", "Act 1884", mean_crease, mean_act),
    ("Begbie", "Act 1884", mean_begbie, mean_act),
]
for a, b, va, vb in pairs:
    sim = cosine_similarity(va, vb)[0, 0]
    print(f"Cosine sim ({a} vs {b}): {sim:.4f}")
```

### Visualizing Embeddings with UMAP

[UMAP](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Uniform_manifold_approximation_and_projection) (Uniform Manifold Approximation and Projection) is a [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction) technique that projects high-dimensional vectors into 2D while preserving local structure. This allows you to visually inspect whether different authors' texts cluster together or apart.

```python
all_vecs = np.vstack(
    embeddings_dict["Crease"]
    + embeddings_dict["Begbie"]
    + embeddings_dict["Act 1884"]
)
labels = (
    ["Crease"] * len(embeddings_dict["Crease"])
    + ["Begbie"] * len(embeddings_dict["Begbie"])
    + ["Act 1884"]
    * len(embeddings_dict["Act 1884"])
)

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
)
proj = reducer.fit_transform(all_vecs)

color_map = {
    "Crease": "#1f77b4",
    "Begbie": "#d62728",
    "Act 1884": "#2ca02c",
}
fig, ax = plt.subplots(figsize=(8, 5))
for author, color in color_map.items():
    mask = [lb == author for lb in labels]
    ax.scatter(
        proj[mask, 0], proj[mask, 1],
        c=color, label=author,
        s=15, alpha=0.8,
    )
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title(
    "UMAP Projection of Stance Embeddings"
)
ax.legend()
plt.tight_layout()
plt.savefig(
    "data/natural-language-inference"
    "-historical-text-01.png",
    dpi=150,
)
plt.show()
```

{% include figure.html filename="data/natural-language-inference-historical-text-01.png" alt="A 2D UMAP projection scatter plot showing legal text embeddings colored by author: Crease (blue), Begbie (red), and Act 1884 (green)" caption="Figure 1: UMAP projection of stance embeddings by author. Crease and Begbie snippets partially overlap, while the Act forms a more distinct cluster." %}

### Investigating Key Sentences

Embeddings provide a quantitative summary, but you should ground those numbers in the actual text. The function below retrieves the sentences most similar to a given author's mean embedding, letting you verify what the model considers representative.

```python
def top_similar_sentences(
    mean_emb, label, n=10
):
    rows = []
    for auth, snippets in act_snippets.items():
        for snippet, emb in zip(
            snippets, embeddings_dict[auth]
        ):
            sim = cosine_similarity(
                emb.reshape(1, -1), mean_emb
            )[0][0]
            rows.append([auth, snippet, sim])
    result = pd.DataFrame(
        rows,
        columns=["Author", "Text", "Similarity"],
    )
    result = result.sort_values(
        "Similarity", ascending=False
    )
    print(
        f"Top {n} sentences closest to"
        f" {label}'s mean embedding:\n"
    )
    for _, row in result.head(n).iterrows():
        txt = textwrap.fill(
            row["Text"], width=78
        )
        print(
            f"Author: {row['Author']}\n"
            f"Text: {txt}\n"
            f"Similarity: "
            f"{row['Similarity']:.4f}\n"
        )
```

```python
top_similar_sentences(mean_crease, "Crease")
```

```python
top_similar_sentences(mean_begbie, "Begbie")
```

```python
top_similar_sentences(mean_act, "Act 1884")
```

## Topic Alignment Analysis

The TF-IDF analysis identified key themes: "labor", "legislation", "license", and "taxation." You can now measure how closely each author's sentences align with these topics by creating topic anchor vectors from descriptive phrases and computing cosine similarity against each snippet embedding.

Rather than building anchors from individual word embeddings (which discard phrasal context), you define each topic with a short descriptive sentence and embed it directly with the same Sentence-BERT model used for the snippets. This produces anchors in the same vector space, making the similarity scores directly comparable.

```python
topic_descriptions = {
    "labor": (
        "Labor, employment, workers,"
        " workforce, economic contribution"
        " of laborers"
    ),
    "legislation": (
        "Legislation, laws, statutes,"
        " legal enactment, parliamentary"
        " authority"
    ),
    "license": (
        "License, permit, fee,"
        " registration, business regulation"
    ),
    "taxation": (
        "Taxation, tax, revenue, duty,"
        " tariff, fiscal policy"
    ),
}

topic_anchors = {
    topic: embed_model.encode(desc)
    for topic, desc in topic_descriptions.items()
}
```

```python
similarity_scores = []
for topic, anchor in topic_anchors.items():
    for author in ['Crease', 'Begbie',
                    'Act 1884']:
        emb_list = embeddings_dict.get(
            author, []
        )
        texts = act_snippets.get(author, [])
        if not emb_list:
            continue
        sims = cosine_similarity(
            anchor.reshape(1, -1),
            np.vstack(emb_list),
        ).flatten()
        for idx, score in enumerate(sims):
            similarity_scores.append({
                'Author': author,
                'Topic': topic,
                'Text': (
                    texts[idx]
                    if idx < len(texts)
                    else ""
                ),
                'Similarity Score': float(score),
            })

similarity_df = pd.DataFrame(
    similarity_scores
)
```

{% include figure.html filename="data/natural-language-inference-historical-text-02.png" alt="Four box plots showing cosine similarity to topic anchors (labor, legislation, license, taxation) by author, with diamond markers for means" caption="Figure 2: Topic similarity distributions by author. Crease shows higher alignment with labor and taxation topics, while Begbie aligns more with license." %}

## Zero-Shot Classification

Zero-shot classification is the core analytical technique of this lesson. It uses a [natural language inference](https://en.wikipedia.org/wiki/Textual_entailment) model to classify text into categories defined at inference time, requiring no labeled training data. This is particularly valuable for historical research, where labeled datasets rarely exist.

### How Zero-Shot NLI Works

NLI models are trained to evaluate pairs of texts: a premise (the input text) and a hypothesis (a candidate label). The model predicts whether the premise *entails* the hypothesis (supports it), *contradicts* it, or is *neutral*. In zero-shot classification, each candidate label is converted into a hypothesis using a template, and the model scores how well the premise entails each hypothesis.

For example, given the premise "The Chinese laborer contributes greatly to the economic development of the province" and the hypothesis "In this text, the author advocates for equal legal treatment of Chinese immigrants," the model would likely assign a high entailment score, classifying the sentence as "Pro."

The key advantage is flexibility: you can define any set of labels without retraining the model; the key risk is that results depend heavily on how you phrase those labels. 

### Designing Effective Labels

Label design is a form of prompt engineering. Vague labels like "positive" or "negative" produce noisy results because the model cannot determine *what* the text is positive or negative about. Labels must specify the exact stance dimension you are measuring.

For this case study, the following labels capture the three positions of interest:

- Pro: "advocates for equal legal treatment of Chinese immigrants compared to white or European settlers, opposing racial discrimination"
- Neutral: "describes or retells the status or treatment of Chinese immigrants without expressing support or opposition to racial inequality, is unrelated to Chinese immigrants, or cannot be classified as either"
- Cons: "justifies or reinforces unequal legal treatment of Chinese immigrants relative to white or European settlers, supporting racially discriminatory policies"

Each label is phrased as a completion of the hypothesis template "In this snippet of a historical legal text, the author {}." This grounds the model in the specific domain and authorial framing of the texts.

A major limitation: results depend heavily on label quality. Labels poorly aligned with the stance categories produce misleading classifications, especially for historical texts whose rhetorical conventions differ from modern usage.

### Setting Up the Pipeline

```python
warnings.filterwarnings("ignore")

model_name = (
    "MoritzLaurer/"
    "deberta-v3-large-zeroshot-v2.0"
)
zero_shot = pipeline(
    "zero-shot-classification",
    model=model_name,
    tokenizer=model_name,
    hypothesis_template=(
        "In this snippet of a historical"
        " legal text, the author {}."
    ),
    device=(
        0 if torch.cuda.is_available()
        else -1
    ),
)

zs_labels = [
    (
        "advocates for equal legal treatment"
        " of Chinese immigrants compared to"
        " white or European settlers,"
        " opposing racial discrimination"
    ),
    (
        "describes or retells the status or"
        " treatment of Chinese immigrants"
        " without expressing support or"
        " opposition to racial inequality,"
        " is unrelated to Chinese immigrants,"
        " or cannot be classified as either"
    ),
    (
        "justifies or reinforces unequal"
        " legal treatment of Chinese"
        " immigrants relative to white or"
        " European settlers, supporting"
        " racially discriminatory policies"
    ),
]

def get_scores(snippet, max_length=512):
    if not snippet or len(snippet.strip()) < 10:
        return {l: 0.33 for l in zs_labels}
    if len(snippet) > max_length * 4:
        snippet = snippet[: max_length * 4]
    out = zero_shot(
        snippet,
        candidate_labels=zs_labels,
        truncation=True,
        max_length=max_length,
    )
    return dict(zip(out["labels"], out["scores"]))
```

### Testing with a Known Example

Before running the pipeline on the full corpus, test it on a passage with a known stance. This paragraph from Sir Chapleau's Royal Commission report discusses Chinese immigration in cautiously economic terms:

> That assuming Chinese immigrants of the laboring class will persist in retaining their present characteristics of Asiatic life, where these are strikingly peculiar and distinct from western, and that the influx will continue to increase, this immigration should be dealt with by Parliament; but no legislation should be such as would give a shock to great interests and enterprises established before any probability that Parliament would interfere with that immigration arose.

```python
chapleau_snippet = (
    "That assuming Chinese immigrants of the"
    " laboring class will persist in retaining"
    " their present characteristics of Asiatic"
    " life, where these are strikingly peculiar"
    " and distinct from western, and that the"
    " influx will continue to increase, this"
    " immigration should be dealt with by"
    " Parliament; but no legislation should be"
    " such as would give a shock to great"
    " interests and enterprises established"
    " before any probability that Parliament"
    " would interfere with that immigration"
    " arose."
)

scores = get_scores(chapleau_snippet)
print("Scores for Chapleau snippet:")
for label, score in scores.items():
    print(f"  {score:.4f}: {label[:50]}...")
```

The model should assign a relatively high "Neutral" score to this passage, which discusses immigration policy without explicitly advocating for or against equal treatment.

### Validating Against Ground Truth

Before interpreting the zero-shot results on the full corpus, it is important to measure the model's accuracy on a labeled sample. A ground truth set of 44 manually labeled snippets (23 contradiction, 18 neutral, 3 entailment) from the corpus provides a benchmark. The model achieves 72.7% accuracy (32 out of 44 correct), with per-class F1 scores (the harmonic mean of precision and recall, where 1.0 is perfect) of 0.75 for contradiction, 0.65 for neutral, and 1.00 for entailment. This is a reasonable baseline for zero-shot classification on nineteenth-century legal text, but the neutral category shows the most confusion, reflecting the difficulty of distinguishing genuinely neutral passages from passages whose stance depends on rhetorical context.

### Sentence-Level Classification

One limitation of transformer models is a fixed token limit (typically 512 tokens). For longer documents, you must split text into smaller units. The sentence approach classifies each sentence individually, capturing fine-grained variation in stance.

Classification of all sentences takes approximately 20 to 40 minutes on CPU. Pre-computed results are provided in `zero_shot_sentence_scores.csv`:

```python
df_scores = pd.read_csv(
    "data/zero_shot_sentence_scores.csv"
)
```

```python
mean_scores = (
    df_scores
    .groupby("Author")[["Pro", "Neutral", "Cons"]]
    .mean()
)
print("Mean scores by author (sentence):")
print(mean_scores.round(4))
```

```python
top_pro = df_scores.nlargest(5, 'Pro')
print("\nTop 5 Pro sentences:\n")
for _, row in top_pro.iterrows():
    txt = textwrap.fill(row['Text'], width=78)
    print(
        f"Author: {row['Author']}\n"
        f"Text: {txt}\n"
        f"Pro: {row['Pro']:.4f}\n"
    )
```

{% include figure.html filename="data/natural-language-inference-historical-text-03.png" alt="Scatter plot of Pro versus Cons zero-shot classification scores colored by author, showing that Act 1884 points cluster toward higher Cons scores" caption="Figure 3: Pro versus Cons classification scores by author (sentence level). The Act clusters toward higher Cons scores, while Crease and Begbie sentences distribute more broadly." %}

### Window-Level Classification

The sentence approach captures variation but loses context. The window approach classifies larger overlapping chunks of text, providing a more holistic stance assessment at the cost of per-sentence nuance.

The windowing function uses the NLI tokenizer to measure token lengths, ensuring each chunk fits within the model's 512-token limit:

```python
nli_tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

def chunk_into_windows(
    text, max_tokens=512, stride=128
):
    sents = sent_tokenize(text)
    windows, current = [], ""
    for sent in sents:
        cand = (
            current + " " + sent
            if current else sent
        )
        n = len(nli_tokenizer.encode(
            cand, add_special_tokens=False
        ))
        if n <= max_tokens:
            current = cand
        else:
            windows.append(current)
            tail = nli_tokenizer.encode(
                current,
                add_special_tokens=False,
            )[-stride:]
            current = (
                nli_tokenizer.decode(tail)
                + " " + sent
            )
    if current:
        windows.append(current)
    return windows
```

Pre-computed windowed scores are provided:

```python
all_scores = pd.read_csv(
    "data/zero_shot_windowed_scores.csv"
)

mean_w = (
    all_scores
    .groupby("Author")[["Pro", "Neutral", "Cons"]]
    .mean()
)
print("Mean scores by author (window):")
print(mean_w.round(4))
```

### Interpreting and Evaluating Results

Both the sentence and window approaches correctly identified the 1884 Chinese Regulation Act as more discriminatory and placed the three authors in the expected relative order. However, the results require careful human evaluation.

Consider this example from Crease's ruling: "...every Chinese is guilty until proved innocent, a provision which fills one conversant with subjects with alarm..." The model classifies this as "Cons" because the surface language describes discriminatory treatment. But in context, Crease is *condemning* the Act's presumption of guilt. This illustrates a fundamental limitation: zero-shot models struggle with complex rhetorical devices where an author describes discriminatory provisions precisely in order to criticize them.

To evaluate zero-shot results responsibly:

1. Examine high-confidence predictions and verify them against the source text
2. Look for systematic misclassifications (e.g., all quotations from the Act within a critique being labeled "Cons")
3. Compare sentence-level and window-level results; disagreements indicate context-sensitive passages
4. Treat the aggregate statistics (mean scores by author) as more reliable than individual sentence scores

## Putting It All Together

The final step combines topic alignment with stance scores. For sentences that appear in both the embedding analysis and the zero-shot classification, you can compute [Pearson correlation coefficients](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) between topic similarity and stance scores:

$$
r = \frac{\text{cov}(X, Y)}{\sigma_X \cdot \sigma_Y}
$$

```python
def clean_text(text):
    text = text.lower()
    return re.sub(r'[^\w\s]', '', text).strip()

similarity_wide = similarity_df.pivot(
    index=['Author', 'Text'],
    columns='Topic',
    values='Similarity Score',
).reset_index()
similarity_wide['key'] = (
    similarity_wide['Text'].apply(clean_text)
)

df_scores['key'] = (
    df_scores['Text'].apply(clean_text)
)

merged_df = similarity_wide.merge(
    df_scores[['key', 'Pro', 'Neutral', 'Cons']],
    on='key',
    how='inner',
)
print(
    f"Merged corpus: {len(merged_df)} sentences"
)
```

```python
topics = [
    'labor', 'legislation',
    'license', 'taxation',
]
results = []
for author in merged_df['Author'].unique():
    sub = merged_df[
        merged_df['Author'] == author
    ]
    for topic in topics:
        for stance in ['Pro', 'Cons']:
            corr = (
                sub[[topic, stance]]
                .corr().iloc[0, 1]
            )
            results.append({
                'Author': author,
                'Topic': topic,
                'Stance': stance,
                'Correlation': round(corr, 4),
            })

corr_df = pd.DataFrame(results)
corr_wide = corr_df.pivot_table(
    index='Author',
    columns=['Topic', 'Stance'],
    values='Correlation',
)
print("\nCorrelation: topic alignment"
      " vs stance by author:")
print(corr_wide)
```

These correlations reveal whether emphasis on certain topics (e.g., labor, taxation) corresponds with more pro- or anti-discriminatory language. Combined with the qualitative evidence from the sentence-level analysis, this provides a multi-faceted view of each author's stance.

### Comparing Lexicon and NLI Results

Following the practice in computational finance of comparing dictionary-based and model-based classifiers, you can now score each sentence with both the lexicon and the NLI model. Where both methods agree, confidence is higher. Where they disagree, the passage likely involves rhetorical complexity — quotation for critique, legal description, or contextual reversal — that merits close reading.

{% include figure.html filename="data/natural-language-inference-historical-text-06.png" alt="Two scatter plots comparing lexicon category scores against NLI classification scores per sentence, showing that many sentences with high NLI Cons scores have zero lexicon hits" caption="Figure 6: Lexicon scores versus NLI scores at the sentence level. The left panel shows that many sentences classified as discriminatory by NLI have zero Exclusionary lexicon hits, indicating the NLI model captures contextual meaning the lexicon misses. The right panel shows weak correlation between Rights-Affirming vocabulary and NLI Pro scores." %}

The comparison reveals that the lexicon and NLI model capture different aspects of stance. Sentences with high NLI Cons scores but zero lexicon hits express discriminatory meaning through phrasing rather than keywords. Sentences where the lexicon flags Exclusionary vocabulary but NLI classifies as Neutral or Pro often involve quotation or description of discriminatory provisions. These disagreements are precisely the passages a historian should examine through close reading.

## Robustness Checks

Computational results from zero-shot NLI should be treated as hypotheses, not conclusions. Three robustness checks help assess how stable the findings are: quote sensitivity analysis, label sensitivity analysis, and bootstrap confidence intervals.

### Quote Sensitivity

Even after removing near-exact quotations of the Act from Crease's text, some paraphrased passages may remain. To test whether residual quotations drive the results, you can apply progressively stricter similarity thresholds and check whether Crease's mean scores change substantially:

```python
crease_sc = df_scores[
    df_scores['Author'] == 'Crease'
].copy()
crease_sc['quote_sim'] = [
    compute_quote_similarity(t, act_sents)
    for t in crease_sc['Text']
]

for threshold in [0.3, 0.4, 0.5, 0.6]:
    filtered = crease_sc[
        crease_sc['quote_sim'] <= threshold
    ]
    n_removed = (
        len(crease_sc) - len(filtered)
    )
    means = filtered[
        ['Pro', 'Neutral', 'Cons']
    ].mean()
    print(
        f"Threshold {threshold:.1f}: "
        f"{n_removed} removed, "
        f"Pro={means['Pro']:.4f} "
        f"Neutral={means['Neutral']:.4f} "
        f"Cons={means['Cons']:.4f}"
    )
```

If the mean scores remain stable across thresholds, the results are not driven by residual Act quotations.

### Label Sensitivity

Zero-shot classification results depend heavily on how candidate labels are phrased. Testing alternative label sets helps determine whether the ranking of authors is an artifact of specific wording or a robust finding:

```python
alt_labels_1 = [
    "supports equal rights for Chinese"
    " immigrants",
    "is neutral or unrelated to Chinese"
    " immigrant rights",
    "supports discriminatory treatment"
    " of Chinese immigrants",
]

alt_labels_2 = [
    "argues that Chinese immigrants"
    " deserve the same legal protections"
    " as other residents",
    "discusses Chinese immigration without"
    " taking a clear legal position for"
    " or against",
    "argues that restricting Chinese"
    " immigrants through law is justified"
    " or necessary",
]
```

If the relative ordering of authors (e.g., Act > Begbie > Crease on "Cons") holds across all three label sets, the finding is more likely to reflect genuine textual patterns rather than label-dependent artifacts.

### Bootstrap Confidence Intervals

With small sample sizes, mean scores can be misleading. Bootstrap resampling provides 95% confidence intervals that quantify the uncertainty in each estimate:

```python
def bootstrap_ci(
    data, n_boot=1000, ci=0.95, seed=42
):
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(
            data, size=len(data), replace=True
        )
        means.append(np.mean(sample))
    lo = np.percentile(
        means, (1 - ci) / 2 * 100
    )
    hi = np.percentile(
        means, (1 + ci) / 2 * 100
    )
    return np.mean(data), lo, hi

for author in ['Crease', 'Begbie',
               'Act 1884']:
    sub = df_scores[
        df_scores['Author'] == author
    ]
    print(f"\n{author} (n={len(sub)}):")
    for stance in ['Pro', 'Neutral', 'Cons']:
        m, lo, hi = bootstrap_ci(
            sub[stance].values
        )
        print(
            f"  {stance}: {m:.4f}"
            f" [{lo:.4f}, {hi:.4f}]"
        )
```

{% include figure.html filename="data/natural-language-inference-historical-text-04.png" alt="Dot-and-whisker plot showing bootstrap 95 percent confidence intervals for Pro, Neutral, and Cons mean scores by author" caption="Figure 4: Bootstrap 95% confidence intervals for mean stance scores. Begbie's wide intervals reflect the smaller sample size (18 snippets versus 83 for Crease)." %}

Wide confidence intervals (especially for Begbie with only 18 snippets) indicate that the point estimates should be interpreted cautiously. Where intervals for different authors overlap on a given stance, the difference between them is not statistically reliable.

### Further Robustness Strategies

The three checks above (quote sensitivity, label sensitivity, and bootstrap confidence intervals) are the minimum recommended for any zero-shot NLI analysis. Depending on the scope of your project, several additional strategies can strengthen confidence in the results:

- Model comparison: re-run classification with a different NLI model (such as `facebook/bart-large-mnli` or `cross-encoder/nli-deberta-v3-base`) and check whether the relative ordering of authors is preserved. If rankings change substantially across models, the finding may be an artifact of one model's training biases rather than a genuine textual pattern.

- Aggregation granularity: compare results at different levels of analysis (sentence, window, full document). Consistent rankings across all levels provide stronger evidence than results that appear only at one granularity.

- Hypothesis template variation: the hypothesis template ("In this snippet of a historical legal text, the author {}") frames how the model interprets each label. Testing alternative templates (such as "The author of this text {}" or "This passage {}") reveals whether the template wording introduces systematic bias.

- Inter-annotator agreement: have two or more domain experts independently label a sample of sentences, then compare the model's classifications against each annotator and against inter-annotator agreement (using metrics like [Cohen's kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa) or [Krippendorff's alpha](https://en.wikipedia.org/wiki/Krippendorff%27s_alpha), which measure the degree of agreement beyond what would be expected by chance). This grounds the model's error rate in human disagreement rather than a single annotator's judgment.

- Temporal sub-sampling: if the corpus spans multiple time periods, classify each period separately and check whether the model's behavior is consistent across decades. Shifts in language conventions over time may affect the model differently for different sub-corpora.

- Permutation testing: randomly shuffle the author labels across sentences and re-compute mean scores. If the observed difference between authors exceeds 95% of the permuted differences, the finding is unlikely to be due to chance.

None of these checks require re-training a model. They can all be implemented with the same zero-shot pipeline demonstrated above, making them accessible additions to any NLI-based historical text analysis.

## Discussion

Returning to the historiographical question: did Crease oppose the 1884 Chinese Regulation Act out of principled concern for Chinese immigrants' rights, or primarily for economic and jurisdictional reasons?

The computational evidence suggests a nuanced answer. The lexicon baseline showed that Crease's texts score higher on ECONOMIC and SANITARY_MEDICAL categories than the other judges, while Begbie's texts carry the highest DEHUMANIZING count (driven by frequent use of "Chinaman" in case rulings — a term that, in the conventions of the period, did not necessarily signal hostility). The embedding-based similarity analysis revealed that Crease and Begbie occupy overlapping regions of semantic space, both distinct from the Act's cluster. Zero-shot NLI classification confirmed that the Act consistently scores highest on discriminatory labels, while Crease and Begbie show more mixed profiles.

Yet the analysis also exposed important limitations. Crease's frequent use of economic vocabulary may reflect his legal strategy — arguing that immigration regulation fell under federal rather than provincial authority — rather than a genuine concern for the economic welfare of Chinese residents. The NLI model cannot distinguish between these motivations because both produce similar surface-level language. Similarly, when Crease quotes the Act to critique it, lexicon-based methods count those words as if they were his own, inflating discriminatory scores unless quotations are carefully removed.

These ambiguities are not failures of the method; they are precisely the kind of interpretive questions that computational analysis is designed to surface. The workflow narrowed the search space from over 80,000 words of legal text to a small set of sentences and patterns that merit close reading. A historian might now focus on the specific passages where Crease's language most closely resembles the Act's, or where Begbie's seemingly protective rulings still employ dehumanizing terminology. The computational pipeline transforms an open-ended question into a tractable set of leads.

## Conclusion

This lesson demonstrated a complete NLP workflow for historical text analysis: from exploratory keyword analysis (TF-IDF) through contextual embeddings (Sentence-BERT), dimensionality reduction (UMAP), zero-shot NLI classification (DeBERTa v2.0), and robustness checks (quote sensitivity, label sensitivity, and bootstrap confidence intervals). The analysis confirmed that Crease and Begbie used semantically similar language when discussing Chinese immigration, while the 1884 Chinese Regulation Act showed a distinctly more discriminatory profile.

The results also highlighted significant limitations. Pre-trained models encode modern semantic associations and may misinterpret historical language, rhetorical devices, and legal conventions. Zero-shot classification is sensitive to label phrasing and cannot reliably distinguish between describing discrimination and endorsing it. The lexicon mismatch between nineteenth-century legal texts and modern training data introduces systematic uncertainty that robustness checks can quantify but not eliminate.

These techniques should be used as discovery tools that generate hypotheses for further investigation, not as substitutes for close reading. The computational workflow narrows the search space and identifies patterns across large corpora, but human expertise remains essential for interpreting what those patterns mean in historical context.

## Further Reading

- Underwood, Ted. *Distant Horizons: Digital Evidence and Literary Change*. Chicago: University of Chicago Press, 2019. An accessible introduction to using computational methods for historical literary analysis.
- Yin, Wenpeng, Jamaal Hay, and Dan Roth. "Benchmarking Zero-shot Text Classification: Datasets, Evaluation, and Entailment Approach." In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*, 3914-23. Hong Kong: Association for Computational Linguistics, 2019. The foundational paper on using NLI for zero-shot text classification.
- Lazer, David, et al. "Computational Social Science." *Science* 323, no. 5915 (2009): 721-23. A landmark article on the promises and pitfalls of computational approaches to social questions.
- Hamilton, William L., Jure Leskovec, and Dan Jurafsky. "Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change." In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, 1489-1501. Berlin: ACL, 2016. Describes the HistWords approach to tracking semantic shifts over time.

## Endnotes

[^1]: *Regina v. Wing Chong*, 1 B.C.R. Pt. II 150 (1885).
[^2]: *Wong Hoy Woon v. Duncan*, 3 B.C.R. 318 (1894).
[^3]: *Regina v. Mee Wah*, 3 B.C.R. 403 (1886).
[^4]: *Regina v. Corporation of Victoria*, 1 B.C.R. Pt. II 331 (1888).
[^5]: *An Act to Regulate the Chinese Population of British Columbia* (S.B.C. 1884, c. 4).
[^6]: Law Society of British Columbia, *The British Columbia Reports: Being Reports of Cases Determined in the Supreme and County Courts and in Admiralty and on Appeal in the Full Court and Divisional Court*, vol. 3 (Victoria, BC: The Province Publishing Company, 1896).
[^7]: Canada, Royal Commission on Chinese Immigration, *Report of the Royal Commission on Chinese Immigration: Report and Evidence* (Ottawa: Printed by order of the Commission, 1885).
[^8]: Paul Thomas, "Courts of Last Resort: The Judicialization of Asian Canadian Politics 1878 to 1913" (paper presented at the Annual Conference of the Canadian Political Science Association, University of Alberta, Edmonton, Canada, June 12-14, 2012), https://cpsa-acsp.ca/papers-2012/Thomas-Paul.pdf.
[^9]: John P.S. McLaren, "The Early British Columbia Supreme Court and the 'Chinese Question': Echoes of the Rule of Law," *Manitoba Law Journal* 20, no. 1 (1991): 107-47, https://www.canlii.org/w/canlii/1991CanLIIDocs168.pdf.
[^10]: Nils Reimers and Iryna Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing* (Hong Kong: Association for Computational Linguistics, 2019), 3982-92, https://doi.org/10.18653/v1/D19-1410.
[^11]: Tina Loo, "Crease, Sir Henry Pering Pellew," in *Dictionary of Canadian Biography*, vol. 13 (University of Toronto/Université Laval, 1994), https://www.biographi.ca/en/bio/crease_henry_pering_pellew_13E.html.
[^12]: David R. Williams, "Begbie, Sir Matthew Baillie," in *Dictionary of Canadian Biography*, vol. 12 (University of Toronto/Université Laval, 1990), https://www.biographi.ca/en/bio/begbie_matthew_baillie_12E.html.
[^13]: Fatemeh Ariai, Joel Mackenzie, and Guido De Martini, "Natural Language Processing for the Legal Domain: A Survey of Tasks, Datasets, Models and Challenges," arXiv:2410.21306 (2025).
[^14]: Marc Alexander, ed., *Historical Thesaurus of English*, 2nd ed. (Glasgow: University of Glasgow, 2020), https://ht.ac.uk/.
[^15]: Jean-Baptiste Michel et al., "Quantitative Analysis of Culture Using Millions of Digitized Books," *Science* 331, no. 6014 (2011): 176-82, https://doi.org/10.1126/science.1199644.
[^16]: EarlyPrint Project, *EarlyPrint: Curating and Exploring Early Printed English* (Northwestern University and Washington University in St. Louis), https://earlyprint.org/.
[^17]: Mark Davies, *Corpus of Historical American English (COHA): 475 Million Words, 1820s–2010s* (Provo, UT: Brigham Young University, 2010–), https://www.english-corpora.org/coha/.
[^18]: Tim Loughran and Bill McDonald, "When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks," *Journal of Finance* 66, no. 1 (2011): 35-65, https://doi.org/10.1111/j.1540-6261.2010.01625.x.
