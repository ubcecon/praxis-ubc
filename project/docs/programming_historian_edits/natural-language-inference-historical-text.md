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
date: YYYY-MM-DD
reviewers:
- Forename Surname
- Forename Surname
editors:
- Laura Alice Chapot
review-ticket: https://github.com/programminghistorian/ph-submissions/issues/699
activity: analyzing
difficulty: 3
topics: [python, distant-reading, machine-learning]
abstract: This lesson demonstrates how to use zero-shot Natural Language Inference (NLI) classification in Python to assess authorial stance in historical legal texts, using nineteenth-century British Columbia court rulings on Chinese immigration as a case study. It covers preparing a corpus (including removing quoted passages with fuzzy string matching), designing classification labels, running sentence- and window-level classification, validating against a hand-labeled ground truth, and applying robustness checks.
mathjax: true
avatar_alt: Visual description of lesson image.
doi: XX.XXXXX/phen0000
---

{% include toc.html %}

## Lesson Goals

This lesson teaches you how to apply [Natural Language Inference (NLI)](https://en.wikipedia.org/wiki/Textual_entailment) techniques to historical documents using Python. NLI allows a model to determine whether a given text entails, contradicts, or is neutral toward a specific historical claim. This framework circumvents the need to build and train custom classifiers for every unique historical debate. Instead of mapping a document to a fixed label, NLI leverages pre-trained semantic reasoning to dynamically assess stances, making it a highly adaptable and powerful tool for historical corpora where labeled training data is unavailable or manual labeling is expensive.

By the end, you will be able to:

- Detect and remove direct quotations using fuzzy string matching
- Extract topically relevant passages from a corpus by keyword
- Apply zero-shot NLI classification to assess stance without labeled data
- Design effective classification labels for historical text analysis
- Validate model outputs against a labeled ground truth set
- Apply robustness checks (quote sensitivity, label sensitivity, bootstrap confidence intervals) to assess result stability
- Critically evaluate NLP results against domain knowledge

The lesson uses nineteenth-century British Columbia court rulings on Chinese immigration as its case study. However, the workflow applies to any historical corpus where you want to computationally assess authorial stance.

The lesson is organized into four stages:

1. **Data preparation** — load the corpus, detect and remove directly quoted passages, and extract the passages that discuss Chinese immigration.
2. **Stance classification** — design labels and run zero-shot NLI at both the sentence and window level.
3. **Evaluation and robustness** — validate predictions against a hand-labeled set and stress-test them with quote, label, and bootstrap checks.
4. **Interpretation** — read the results critically and adapt the workflow to other corpora.

## Prerequisites

You will need intermediate Python experience: working with pandas, writing functions, and using pip. If you are newer to Python, start with the [_Programming Historian_'s Introduction to Python](https://programminghistorian.org/en/lessons/introduction-and-installation).

Python 3.10 or later is required, along with at least 8GB of RAM. A GPU or iGPU is not required but will substantially speed up model inference.

<div class="alert alert-warning">
This lesson uses transformer models and needs at least 8GB of RAM. Running every zero-shot step from scratch can take 60 to 90 minutes on CPU. Pre-computed CSV outputs are included so you can complete the lesson without full recomputation.

If you have NVIDIA CUDA, PyTorch can use it automatically. On some Windows systems with AMD or Intel integrated GPUs, <code>torch-directml</code> may help, but stability varies by model. The notebook is written to run safely on CPU first, then use acceleration where stable.
</div>

## Software and Setup

Install all required Python packages:

```bash
pip install pandas numpy matplotlib \
  seaborn nltk spacy scipy \
  transformers torch tqdm
```

For AMD or Intel integrated GPU acceleration on Windows, also install:

```bash
pip install torch-directml
```

Download NLTK data and the spaCy language model:

```python
import nltk
nltk.download('punkt')
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
- DeBERTa NLI: `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` (Hugging Face)

Load the required libraries:

```python
import pandas as pd
import numpy as np
import difflib
from nltk import sent_tokenize
import spacy
from transformers import AutoTokenizer, pipeline
import torch
import warnings
```

### Downloading the Data

Download the lesson data files from the [_Programming Historian_ repository](https://github.com/programminghistorian/ph-submissions/assets/natural-language-inference-historical-text/data.zip). Create a `data/` directory in your working folder and place all files there. The dataset includes:

- `data/core/metadata_cleaned.csv` -- a table listing the ten source documents with author, group, and type metadata
- Thirteen `.txt` files in `data/texts/` -- the OCR-transcribed historical texts (legal rulings, the 1884 Chinese Regulation Act, and Royal Commission reports)
- `data/core/labelled_snippets.csv` -- 45 hand-labelled sentence excerpts used for ground truth validation
- `data/texts/quotations_removed/` -- versions of Crease's texts with direct quotations of the Act removed

## Case Study: Chinese Immigration Law in British Columbia

While the techniques demonstrated in this lesson are general-purpose, you will go through a case study that provides concrete material to work with.

**A note on language**: This lesson reproduces historical racist terminology from the source documents (*Chinaman*, *Chinamen*, *coolie*, *heathen*, and *alien* in its nineteenth-century legal context). These terms appear in direct quotations, keyword lists, and the code cells because accurate computational analysis of nineteenth-century anti-Chinese legislation requires working with the source vocabulary. They are presented here as historical evidence.

The *1884 Chinese Regulation Act*[^1] in British Columbia (a province on the Pacific coast of Canada) was provincial legislation targeting Chinese residents, part of a broader wave of anti-Chinese laws across western North America in the late nineteenth century. It was challenged and declared unconstitutional in the 1885 case of *R v. Wing Chong* by [Henry Pering Pellew Crease](https://www.biographi.ca/en/bio/crease_henry_pering_pellew_13E.html), a judge on the Supreme Court of British Columbia.[^2] Justice Crease struck down the legislation on economic grounds, finding that it infringed on federal authority over immigration, trade, commerce, and taxation. In Commonwealth legal naming convention, *R* (or *Regina*, Latin for 'the Queen') denotes a criminal or constitutional case brought by the Crown against a private party.

However, Crease was not considered straightforwardly sympathetic to Chinese immigrants. Historian Tina Loo notes that he displayed mistrust toward Chinese residents, referred to them as "North American Chinamen," and feared they would "rule the country and job its offices."[^3] The apparent inconsistency between Crease’s rhetoric and his political position raises a question difficult to answer through selective quotation alone: how did he actually discuss Chinese immigrants across his broader body of writing? And what methods can we use to better understand whether his opposition to the Act was rooted in principled objections to discrimination, or in a belief that Chinese immigrant labor was necessary for economic development?

To explore this question computationally, you will compare the language of Crease's rulings with two reference points: the discriminatory Act itself, and Justice [Matthew Baillie Begbie](https://www.biographi.ca/en/bio/begbie_matthew_baillie_12E.html),[^4] the first Chief Justice of British Columbia. Unlike Crease, historical accounts describe Begbie as protective of marginalized peoples, including Chinese immigrants.[^5][^6] Begbie struck down discriminatory municipal by-laws in Victoria targeting Chinese-owned businesses in the 1888 case of *R v. Victoria*.[^7]

The corpus consists of ten digitized texts: legal rulings (*R v. Wing Chong*,[^2] *Wong Hoy Woon v. Duncan*,[^8] *R v. Mee Wah*,[^9] *R v. Victoria*[^7]), the *1884 Chinese Regulation Act*, and reports from the 1884 Royal Commission on Chinese Immigration.[^10] The texts were converted from archival scans to machine-readable format using [Optical Character Recognition (OCR)](https://en.wikipedia.org/wiki/Optical_character_recognition). Direct quotes of the Act within Crease's ruling were identified using fuzzy string matching and removed so they do not contaminate the analysis of his own language (this process is described in the next section).

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
df = pd.read_csv("data/core/metadata_cleaned.csv")

ACT_LABEL = "Regulation Act"

df["group"] = "Other"
df.loc[df["author"] == "Crease", "group"] = "Crease"
df.loc[df["author"] == "Begbie", "group"] = "Begbie"
df.loc[df["type"] == "act", "group"] = ACT_LABEL
```

Grouping documents by author lets you compare the language of Crease, Begbie, and the Act directly. Check how many documents fall into each group before going further:

```python
df["group"].value_counts()
```

### Detecting and Removing Direct Quotations

Crease's ruling in *R v. Wing Chong* quotes passages from the 1884 Chinese Regulation Act verbatim. If these quoted passages remain in the corpus, the NLI model will classify them as language attributable to Crease, when in fact they are the Act's own words that Crease cited. This is not a peripheral cleanup step: stance classification asks "what does *this author* say," and a passage Crease quotes from the Act in order to strike it down would be scored as if Crease himself wrote it. Quotation removal is therefore a prerequisite for the analysis, not an optional refinement, which is why it belongs here in data preparation rather than among the analytical methods.

The approach uses fuzzy string matching via Python's [`difflib.SequenceMatcher`](https://docs.python.org/3/library/difflib.html), which computes a similarity ratio between two strings based on the longest contiguous matching subsequences. Exact string matching would be too weak here: the texts were produced by OCR, so a sentence Crease quotes from the Act and the same sentence in the Act itself rarely match character-for-character. Small transcription differences (a misread letter, a dropped word, inconsistent punctuation) would defeat an exact-match filter while leaving the quotation in place. Fuzzy matching tolerates this noise by scoring *how similar* two sentences are rather than demanding they be identical. For each sentence in Crease's text, you compute its similarity to every sentence in the Act and retain the highest score:

```python
nlp = spacy.load("en_core_web_sm")

regulation_act_text = df.loc[df["type"] == "act", "text"].iloc[0]
regulation_act_sents = [
    s.text.strip()
    for s in nlp(regulation_act_text).sents
    if len(s.text.strip()) > 20
]

crease_orig_path = "data/texts/analytic/Regina_V_Wing_Chong.txt"

with open(crease_orig_path, encoding="utf-8") as f:
    crease_orig = f.read()

def compute_quote_similarity(sent, reference_sents):
    best = 0.0
    s_lower = sent.lower()

    for ref in reference_sents:
        ratio = difflib.SequenceMatcher(None, s_lower, ref.lower()).ratio()
        if ratio > best:
            best = ratio

    return best
```

The threshold you apply to these similarity scores is a trade-off. A threshold of 0.6 catches near-exact quotes (accounting for OCR errors) while leaving Crease's own paraphrases intact; a looser threshold of 0.4 also catches loose paraphrases but risks discarding original sentences that merely share legal vocabulary with the Act. Set it too high and quoted material survives to contaminate the stance scores; set it too low and you erase the author's own language. The `act_quote_sentences_removed` column in the metadata records how many sentences were removed from each document at the 0.6 threshold, and the cleaned texts are saved under `data/texts/quotations_removed/`. The same `compute_quote_similarity` function is reused later as a robustness check (see [Quote Sensitivity](#quote-sensitivity)), where you sweep across thresholds to confirm that residual quotations are not driving the results. For your own analyses, experiment with different thresholds to see which one best separates quotation from original language in your corpus.

### Identifying Relevant Passages

Zero-shot classification is computationally expensive, so rather than scoring every sentence in every document, you focus the analysis on the passages that actually discuss Chinese immigration. The strategy is to split each author's texts into sentences and keep those that mention immigration-related keywords. These keyword snippets become the unit of analysis for the sentence-level classifier, while the full documents are reused for window-level classification later.

```python
crease_cases = df[
    (df['author'] == 'Crease') & (df['type'] == 'case')
]['text'].tolist()
begbie_cases = df[
    (df['author'] == 'Begbie') & (df['type'] == 'case')
]['text'].tolist()
regulation_act_texts = df[df['type'] == 'act']['text'].tolist()

corpus_by_author = {
    'Crease': crease_cases,
    'Begbie': begbie_cases,
    ACT_LABEL: regulation_act_texts,
}

keywords = [
    "chinese", "china", "chinaman",
    "chinamen", "immigrant", "immigrants",
    "alien", "aliens", "immigration",
]

warnings.filterwarnings("ignore")

keyword_snippets = {}
for author, texts in corpus_by_author.items():
    snippets = []
    for text in texts:
        for sentence in nlp(text).sents:
            sentence_text = sentence.text.strip()
            sentence_lower = sentence_text.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                snippets.append(sentence_text)
    keyword_snippets[author] = snippets
```

Keyword filtering is deliberately simple and high-recall: it casts a wide net so that few relevant passages are missed, and the NLI model does the discriminating work of judging stance later. The `corpus_by_author` dictionary is kept alongside the snippets because the window-level analysis needs the full document text, not just the extracted sentences.

## Stance Classification with Zero-Shot NLI

Zero-shot classification is the core analytical technique of this lesson. It uses a [natural language inference](https://en.wikipedia.org/wiki/Textual_entailment) model to classify text into categories defined at inference time, requiring no labeled training data. This is particularly valuable for historical research, where labeled datasets rarely exist.

### Why NLI?

A natural first instinct is to measure stance by counting words: tally discriminatory or rights-affirming terms and compare authors. This is the logic behind domain-specific lexicons such as the Loughran-McDonald dictionary in financial text analysis, which counts curated word lists tailored to a domain rather than relying on general-purpose sentiment polarity.[^11] But keyword counting has a decisive blind spot for our question: it cannot tell the difference between a judge who *uses* a discriminatory word and one who *quotes it in order to condemn it*. The word "alien" counts the same whether the Act imposes it or Crease attacks it, and stance-bearing phrases such as "fills one with alarm" or "rule the country" use common words that no lexicon would flag. Stance lives in how words are combined and framed, not in just their presense, which is exactly what Natural Language Inference is built to read.

NLI is a practical middle ground for historical corpora. Supervised models need large labeled datasets that most historians do not have, and lexicon or topic methods often miss stance direction.[^18][^19] NLI instead scores whether a passage supports researcher-defined hypotheses, so you can run three-way stance classification without retraining.[^20]

### How Zero-Shot NLI Works

NLI models are trained to evaluate pairs of texts: a premise (the input text) and a hypothesis (a candidate label). The model predicts whether the premise *entails* the hypothesis (supports it), *contradicts* it, or is *neutral*. In zero-shot classification, each candidate label is converted into a hypothesis using a template, and the model scores how well the premise entails each hypothesis.

For example, given the premise "The treaties I have quoted between Great Britain and China, binding on the Dominion and on us in British Columbia, secure to the Chinese, just as the treaties between Great Britain and other foreign countries secure to other foreigners, the same rights in regard to the equality of taxation which I have described as being enjoyed by citizens of this country." and the hypothesis "In this text, the author advocates for equal legal treatment of Chinese immigrants", the model would likely assign a high entailment score, classifying the sentence as "Pro".

The key advantage is flexibility: you can define any set of labels without retraining the model; the key risk is that results depend heavily on how you phrase those labels.

### Choosing a Model for Historical Text

Model selection is a critical decision in any NLP pipeline, especially for historical texts. Two checks matter most: whether the training data is close to your corpus, and whether the model was designed for your task.

This lesson uses [DeBERTa NLI (v2.0)](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) for zero-shot classification. It is tuned for entailment tasks and performs strongly when labels are expressed as explicit hypotheses.[^12]

When choosing models for your own historical corpus, consider:

- Does the model's training data overlap with your domain? A general-purpose model may lack specialized vocabulary, while a domain-specific model trained on modern legal text may not understand nineteenth-century usage of terms like "alien."[^13]
- Is the model designed for your task? Use NLI-fine-tuned models for zero-shot classification rather than general-purpose language models.
- Test with known examples. Pass excerpts where you already know the expected result and check whether the model's output aligns with your domain knowledge.

### The Lexicon Mismatch Problem

A central challenge is lexical drift: words in nineteenth-century legal writing can carry meanings that differ from modern usage. OCR noise adds another layer of uncertainty. There is no complete fix, but you can reduce risk by:

- Verify that key terms in your corpus appear in the model's vocabulary
- Compare model outputs against passages where the expected stance is clear
- Treat all computational results as hypotheses that require human validation, not as conclusions

### Designing Effective Labels

Label design is a form of prompt engineering. Vague labels like "positive" or "negative" produce noisy results because the model cannot determine *what* the text is positive or negative about. Labels must specify the exact stance dimension you are measuring.

For this case study, the following labels capture the three positions of interest:

- Pro: "advocates for equal legal treatment of Chinese immigrants compared to white or European settlers, opposing racial discrimination"
- Neutral: "describes or retells the status or treatment of Chinese immigrants without expressing support or opposition to racial inequality, is unrelated to Chinese immigrants, or cannot be classified as either"
- Cons: "justifies or reinforces unequal legal treatment of Chinese immigrants relative to white or European settlers, supporting racially discriminatory policies"

Each label is phrased as a completion of the hypothesis template "In this snippet of a historical legal text, the author {}." This grounds the model in the specific domain and authorial framing of the texts.

A major limitation is results depend heavily on label quality. Labels poorly aligned with the stance categories produce misleading classifications, especially for historical texts whose rhetorical conventions differ from modern usage.

### Setting Up the Pipeline

The lesson presents the pipeline in its simplest form. The notebook demo keeps the extra environment controls needed for repeatable execution.

```python
warnings.filterwarnings("ignore")
model_name = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
zero_shot = pipeline(
    "zero-shot-classification",
    model=model_name,
    tokenizer=model_name,
    hypothesis_template="In this snippet of a historical legal text, the author {}.",
    device=0 if torch.cuda.is_available() else -1,
)

zs_labels = [
    (
        "advocates for equal legal treatment of Chinese immigrants "
        "compared to white or European settlers, opposing racial "
        "discrimination"
    ),
    (
        "describes or retells the status or treatment of Chinese "
        "immigrants without expressing support or opposition to racial "
        "inequality, is unrelated to Chinese immigrants, or cannot be "
        "classified as either"
    ),
    (
        "justifies or reinforces unequal legal treatment of Chinese "
        "immigrants relative to white or European settlers, supporting "
        "racially discriminatory policies"
    ),
]

def get_scores(snippet, labels):
    out = zero_shot(
        snippet,
        candidate_labels=labels,
        truncation=True,
    )
    raw = dict(zip(out["labels"], out["scores"]))
    return {label: raw.get(label, 0.0) for label in labels}

SCORE_COLS = ['Pro', 'Neutral', 'Cons']

def score_texts(texts_by_author, labels):
    rows = []
    for author, texts in texts_by_author.items():
        for text in texts:
            score_map = get_scores(text, labels=labels)
            rows.append(
                {
                    'Author': author,
                    'Text': text,
                    'Pro': score_map[labels[0]],
                    'Neutral': score_map[labels[1]],
                    'Cons': score_map[labels[2]],
                }
            )
    return pd.DataFrame(rows)
```

The notebook demo keeps the pre-computed load path for faster execution, but the lesson shows the direct calculation steps.

### Sentence-Level Classification

One limitation of transformer models is a fixed token limit (typically 512 tokens). For longer documents, you must split text into smaller units. The sentence approach classifies each sentence individually, capturing fine-grained variation in stance.

Classification of all sentences takes approximately 20 to 40 minutes on CPU. The lesson shows the full scoring step directly, while the notebook demo loads the saved CSV.

```python
sentence_scores_path = "data/results/zero_shot_sentence_scores.csv"
df_scores = score_texts(keyword_snippets, labels=zs_labels)
df_scores.to_csv(sentence_scores_path, index=False)

sentence_summary = df_scores.groupby("Author")[SCORE_COLS].mean().round(4)
```

{% include figure.html filename="en-or-natural-language-inference-historical-text-01.png" alt="Scatter plot of Pro versus Cons zero-shot classification scores colored by author, showing that Regulation Act points cluster toward higher Cons scores" caption="Figure 1. Pro versus Cons classification scores by author (sentence level). The Regulation Act clusters toward higher Cons scores, while Crease and Begbie sentences distribute more broadly." %}

### Window-Level Classification

The sentence approach captures variation but loses context. The window approach classifies larger overlapping chunks of text, providing a more holistic stance assessment at the cost of per-sentence nuance.

The windowing function uses the NLI tokenizer to measure token lengths, ensuring each chunk fits within the model's 512-token limit:

```python
nli_tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

def chunk_into_windows(text, max_tokens=512, stride=128):
    sents = sent_tokenize(text)
    windows, current = [], ""
    for sent in sents:
        cand = current + " " + sent if current else sent
        n = len(nli_tokenizer.encode(cand, add_special_tokens=False))
        if n <= max_tokens:
            current = cand
        else:
            windows.append(current)
            tokens = nli_tokenizer.encode(current, add_special_tokens=False)
            tail = tokens[-stride:]
            current = nli_tokenizer.decode(tail) + " " + sent
    if current:
        windows.append(current)
    return windows
```

The same pattern works for larger windows. The notebook demo loads the saved CSV, but the lesson shows the direct calculation:

```python
window_scores_path = "data/results/zero_shot_windowed_scores.csv"
window_texts_by_author = {
    author: [
        window_text
        for doc_text in docs
        for window_text in chunk_into_windows(doc_text)
    ]
    for author, docs in corpus_by_author.items()
}
window_scores_df = score_texts(window_texts_by_author, labels=zs_labels)
window_scores_df.to_csv(window_scores_path, index=False)

window_summary = window_scores_df.groupby("Author")[SCORE_COLS].mean().round(4)
```

## Evaluation and Robustness

Computational results from zero-shot NLI should be treated as hypotheses, not conclusions. Unlike supervised models evaluated on held-out test sets, zero-shot classifiers carry no built-in accuracy guarantee for a new domain. This section first measures performance against a hand-labeled set, then applies three robustness checks — quote sensitivity, label sensitivity, and bootstrap confidence intervals — to assess how stable the findings are.

### Validating Against Ground Truth

Before interpreting zero-shot results on the full corpus, it is important to measure performance on a labeled sample that matches the task definition. The evaluation set used here contains 45 manually labeled snippets balanced across the three pipeline labels (Pro, Neutral, Cons), with representation from Act text, Crease, Begbie, and Commission material. This design evaluates the same three-way classification problem used in the analysis pipeline, rather than a separate single-hypothesis entailment task.

The evaluation reports overall accuracy, per-class precision/recall/F1 (the harmonic mean of precision and recall, where 1.0 is perfect), and a majority-class baseline. Reporting the baseline is essential: if a trivial classifier can perform well by always predicting one class, apparent gains in accuracy may be misleading. The notebook also reports per-author breakdowns so you can see whether performance is concentrated in one source type or generalizes across legal voices.

In this run, overall accuracy on the 45-sentence set is 0.667 (30/45), compared with a majority-class baseline of 0.333. Per-class F1 scores are 0.500 (Pro), 0.686 (Neutral), and 0.743 (Cons). Per-author accuracy is highest for Commission snippets (1.000), followed by Crease (0.733) and the Regulation Act (0.700), and lower for Begbie (0.467), which is consistent with the rhetorical complexity discussed below.

For interpretive tasks, this level of performance is usable but not definitive. Treat these scores as decision support for close reading, not automated ground truth.[^21][^22][^23][^24]

### Quote Sensitivity

Even after removing near-exact quotations of the Act from Crease's text, some paraphrased passages may remain. To test whether residual quotations drive the results, you can apply progressively stricter similarity thresholds and store the filtered means for comparison:

```python
crease_sc = df_scores[df_scores['Author'] == 'Crease'].copy()
crease_sc['quote_sim'] = [
    compute_quote_similarity(t, regulation_act_sents)
    for t in crease_sc['Text']
]

quote_sensitivity = []
for threshold in [0.3, 0.4, 0.5, 0.6]:
    filtered = crease_sc[crease_sc['quote_sim'] <= threshold]
    means = filtered[['Pro', 'Neutral', 'Cons']].mean()
    quote_sensitivity.append(
        {
            'threshold': threshold,
            'removed': len(crease_sc) - len(filtered),
            'Pro': means['Pro'],
            'Neutral': means['Neutral'],
            'Cons': means['Cons'],
        }
    )

quote_sensitivity = pd.DataFrame(quote_sensitivity)
```

If the mean scores remain stable across thresholds, the results are not driven by residual Act quotations.

### Label Sensitivity

Zero-shot classification results depend heavily on how candidate labels are phrased. Testing alternative label sets helps determine whether the ranking of authors is an artifact of specific wording or a robust finding. The notebook demo loads a saved summary, but the lesson shows the direct comparison step:

```python
alt_labels_1 = [
    "supports equal rights for Chinese immigrants",
    "is neutral or unrelated to Chinese immigrant rights",
    "supports discriminatory treatment of Chinese immigrants",
]

alt_labels_2 = [
    "argues that Chinese immigrants deserve the same legal protections as other residents",
    "discusses Chinese immigration without taking a clear legal position for or against",
    "argues that restricting Chinese immigrants through law is justified or necessary",
]

label_sets = {
    "Primary": zs_labels,
    "Alt-short": alt_labels_1,
    "Alt-legal": alt_labels_2,
}

sample = df_scores.groupby("Author", group_keys=False).head(2)
sample_texts = {
    author: sub["Text"].fillna("").tolist()
    for author, sub in sample.groupby("Author")
}

label_sens_df = pd.concat(
    [
        score_texts(sample_texts, labels=labels)
        .groupby("Author")[SCORE_COLS]
        .mean()
        .reset_index()
        .assign(LabelSet=set_name)
        [["LabelSet", "Author", "Pro", "Neutral", "Cons"]]
        for set_name, labels in label_sets.items()
    ],
    ignore_index=True,
)

label_sens_df.to_csv("data/results/label_sensitivity_summary.csv", index=False)
```

If the relative ordering of authors (e.g., Regulation Act > Begbie > Crease on "Cons") holds across all three label sets, the finding is more likely to reflect genuine textual patterns rather than label-dependent artifacts.

### Bootstrap Confidence Intervals

With small sample sizes, mean scores can be misleading. Bootstrap resampling provides 95% confidence intervals that quantify the uncertainty in each estimate:

```python
def bootstrap_ci(data, n_boot=1000, ci=0.95, seed=42):
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

bootstrap_rows = []
for author in ['Crease', 'Begbie', ACT_LABEL]:
    sub = df_scores[df_scores['Author'] == author]
    for stance in ['Pro', 'Neutral', 'Cons']:
        m, lo, hi = bootstrap_ci(
            sub[stance].values
        )
        bootstrap_rows.append(
            {
                'Author': author,
                'Stance': stance,
                'Mean': m,
                'CI_low': lo,
                'CI_high': hi,
            }
        )

bootstrap_summary = pd.DataFrame(bootstrap_rows)
```

{% include figure.html filename="en-or-natural-language-inference-historical-text-02.png" alt="Dot-and-whisker plot showing bootstrap 95 percent confidence intervals for Pro, Neutral, and Cons mean scores by author" caption="Figure 2. Bootstrap 95% confidence intervals for mean stance scores. Begbie's wide intervals reflect the smaller sample size (18 snippets versus 83 for Crease)." %}

Wide confidence intervals (especially for Begbie with only 18 snippets) indicate that the point estimates should be interpreted cautiously. Where intervals for different authors overlap on a given stance, the difference between them is not statistically reliable.

## Interpreting Results and Adapting the Workflow

### What NLI Can and Cannot Do

Both sentence and window approaches identify the Regulation Act as the most discriminatory source. However, interpretation requires careful human evaluation and explicit treatment of model uncertainty.

Consider this example from Crease's ruling: "...every Chinese is guilty until proved innocent, a provision which fills one conversant with subjects with alarm..." The model may classify this as "Cons" because the sentence contains discriminatory language. In context, however, Crease is *condemning* the law. This pattern appears repeatedly in Begbie as well and forms a central interpretive issue in this workflow.

Linguists and discourse analysts have documented what may be called *quotation-induced stance reversal*: when a speaker quotes another's words to criticize them, surface-level analysis attributes the quoted stance to the speaker.[^25] Sentence-level NLI is particularly vulnerable to this because the model reads the discriminatory words without the surrounding argumentative frame that signals condemnation. Rights-protective legal judgments can therefore receive high "Cons" scores at the sentence level when judges quote or describe discriminatory rules in order to reject them.

To address this, window-level aggregates serve as the primary summary and sentence-level results serve as granular diagnostics. Confidence-aware summaries (filtering rows whose maximum label score falls below 0.5 and computing confidence-weighted means) further reduce the influence of ambiguous sentences.

To evaluate zero-shot results responsibly:

1. Examine high-confidence predictions and verify them against the source text
2. Look for systematic misclassifications (e.g., all quotations from the Act within a critique being labeled "Cons")
3. Compare sentence-level and window-level results; disagreements indicate context-sensitive passages
4. Treat window-level and confidence-aware aggregates as more reliable than any single sentence score

Returning to the historiographical question, the results are mixed rather than binary. The Regulation Act is consistently most discriminatory, while Crease and Begbie show overlapping but internally varied profiles. This supports a cautious interpretation: legal argument, quotation, and rhetorical framing all shape the scores. The method is most useful for prioritizing passages for close reading, not for replacing interpretation.

### Adapting to Your Own Corpus

The case study is specific, but the workflow is general. To adapt it to a new corpus, a researcher must make five decisions:

1. Select and digitize the corpus
2. Define a keyword list that identifies the thematic focus
3. Design classification labels that name the specific stance dimensions of interest
4. Choose a pre-trained model whose training domain approximates the target corpus
5. Assemble a small labeled sample of even 30 to 50 sentences to measure and report classification accuracy

Each of these decisions shapes what the pipeline can and cannot reveal, and each warrants explicit justification in any publication that uses these methods.

### Digital Resources for Historical Semantics

These resources help you check historical usage before final interpretation:

- The [Historical Thesaurus of English](https://ht.ac.uk/) traces when words acquired or lost specific senses through dated attestations — for example, confirming that "alien" carried its legal sense throughout the nineteenth century.[^14]
- [Google Books Ngram Viewer](https://books.google.com/ngrams) charts word frequencies across centuries of digitized books, revealing where historical and modern usage patterns diverge.[^15]
- [EarlyPrint](https://earlyprint.org/) provides linguistically annotated early English print (1473 to the early 1700s) with tools for handling archaic spelling and OCR artifacts.[^16]
- The [Corpus of Historical American English (COHA)](https://www.english-corpora.org/coha/) contains 475 million words from the 1820s to the 2010s, searchable by decade and genre.[^17]

These tools cannot remove model bias, but they help you design better labels and spot likely failure points.

**TODO: add guidance on adapting this workflow to other languages — multilingual NLI models (e.g. mDeBERTa, XLM-R), translating the candidate labels and hypothesis template, and non-English historical-semantics resources. This subsection is a placeholder; do not treat it as written.**

## Conclusion

This lesson demonstrated a full, reproducible workflow: preparing a corpus, removing quoted passages with fuzzy matching, scoring stance with zero-shot DeBERTa NLI, and stress-testing the results with robustness checks. The core finding is stable across methods: the Regulation Act is the clearest discriminatory source, while judicial texts are more ambiguous and context-dependent. Use these outputs as structured evidence for historical interpretation, not as final verdicts.

Ultimately, deploying Natural Language Inference within historical research does not supplant traditional hermeneutics; rather, it significantly scales the historian's analytical capacity. While transformer-based models can efficiently parse vast archives to surface latent ideological alignments or textual contradictions, they remain inherently agnostic to the temporally bound, cultural nuances of historical discourse. Consequently, NLI should be treated as a sophisticated heuristic mechanism. It excels at directing researchers toward critical junctures of conflict or consensus within a corpus, but the final burden of interpretation, contextualization, and causal inference remains firmly with the human scholar.

## Further Reading

- Underwood, Ted. *Distant Horizons: Digital Evidence and Literary Change*. Chicago: University of Chicago Press, 2019. An accessible introduction to using computational methods for historical literary analysis.
- Yin, Wenpeng, Jamaal Hay, and Dan Roth. "Benchmarking Zero-shot Text Classification: Datasets, Evaluation, and Entailment Approach." In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*, 3914-23. Hong Kong: Association for Computational Linguistics, 2019. The foundational paper on using NLI for zero-shot text classification.
- Lazer, David, et al. "Computational Social Science." *Science* 323, no. 5915 (2009): 721-23. A landmark article on the promises and pitfalls of computational approaches to social questions.
- Hamilton, William L., Jure Leskovec, and Dan Jurafsky. "Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change." In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, 1489-1501. Berlin: ACL, 2016. Describes the HistWords approach to tracking semantic shifts over time.

## Endnotes

[^1]: *An Act to Regulate the Chinese Population of British Columbia* (S.B.C. 1884, c. 4).
[^2]: *Regina v. Wing Chong*, 1 B.C.R. Pt. II 150 (1885).
[^3]: Tina Loo, "Crease, Sir Henry Pering Pellew," in *Dictionary of Canadian Biography*, vol. 13 (University of Toronto/Université Laval, 1994), https://www.biographi.ca/en/bio/crease_henry_pering_pellew_13E.html.
[^4]: David R. Williams, "Begbie, Sir Matthew Baillie," in *Dictionary of Canadian Biography*, vol. 12 (University of Toronto/Université Laval, 1990), https://www.biographi.ca/en/bio/begbie_matthew_baillie_12E.html.
[^5]: Paul Thomas, "Courts of Last Resort: The Judicialization of Asian Canadian Politics 1878 to 1913" (paper presented at the Annual Conference of the Canadian Political Science Association, University of Alberta, Edmonton, Canada, June 12-14, 2012), https://cpsa-acsp.ca/papers-2012/Thomas-Paul.pdf.
[^6]: John P.S. McLaren, "The Early British Columbia Supreme Court and the 'Chinese Question': Echoes of the Rule of Law," *Manitoba Law Journal* 20, no. 1 (1991): 107-47, https://www.canlii.org/w/canlii/1991CanLIIDocs168.pdf.
[^7]: *Regina v. Corporation of Victoria*, 1 B.C.R. Pt. II 331 (1888).
[^8]: *Wong Hoy Woon v. Duncan*, 3 B.C.R. 318 (1894).
[^9]: *Regina v. Mee Wah*, 3 B.C.R. 403 (1886).
[^10]: Canada, Royal Commission on Chinese Immigration, *Report of the Royal Commission on Chinese Immigration: Report and Evidence* (Ottawa: Printed by order of the Commission, 1885).
[^11]: Tim Loughran and Bill McDonald, "When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks," *Journal of Finance* 66, no. 1 (2011): 35-65, https://doi.org/10.1111/j.1540-6261.2010.01625.x.
[^12]: Moritz Laurer, Wouter van Atteveldt, Andreu Casas, and Kasper Welbers, "Less is More: Optimal Dataset Size for NLI Models," arXiv:2109.09703 (2023).
[^13]: Fatemeh Ariai, Joel Mackenzie, and Guido De Martini, "Natural Language Processing for the Legal Domain: A Survey of Tasks, Datasets, Models and Challenges," arXiv:2410.21306 (2025).
[^14]: Marc Alexander, ed., *Historical Thesaurus of English*, 2nd ed. (Glasgow: University of Glasgow, 2020), https://ht.ac.uk/.
[^15]: Jean-Baptiste Michel et al., "Quantitative Analysis of Culture Using Millions of Digitized Books," *Science* 331, no. 6014 (2011): 176-82, https://doi.org/10.1126/science.1199644.
[^16]: EarlyPrint Project, *EarlyPrint: Curating and Exploring Early Printed English* (Northwestern University and Washington University in St. Louis), https://earlyprint.org/.
[^17]: Mark Davies, *Corpus of Historical American English (COHA): 475 Million Words, 1820s–2010s* (Provo, UT: Brigham Young University, 2010–), https://www.english-corpora.org/coha/.
[^18]: Bing Liu, *Sentiment Analysis and Opinion Mining* (San Rafael, CA: Morgan & Claypool, 2012).
[^19]: David M. Blei, Andrew Y. Ng, and Michael I. Jordan, "Latent Dirichlet Allocation," *Journal of Machine Learning Research* 3 (2003): 993-1022.
[^20]: Wenpeng Yin, Jamaal Hay, and Dan Roth, "Benchmarking Zero-shot Text Classification: Datasets, Evaluation, and Entailment Approach," in *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing* (Hong Kong: Association for Computational Linguistics, 2019), 3914-23.
[^21]: Ron Artstein and Massimo Poesio, "Inter-Coder Agreement for Computational Linguistics," *Computational Linguistics* 34, no. 4 (2008): 555-96, https://doi.org/10.1162/coli.07-034-R2.
[^22]: Klaus Krippendorff, *Content Analysis: An Introduction to Its Methodology*, 4th ed. (Los Angeles: Sage, 2018).
[^23]: Rion Snow, Brendan O'Connor, Daniel Jurafsky, and Andrew Y. Ng, "Cheap and Fast — But Is It Good? Evaluating Non-Expert Annotations for Natural Language Tasks," in *Proceedings of the 2008 Conference on Empirical Methods in Natural Language Processing* (Honolulu: Association for Computational Linguistics, 2008), 254-63.
[^24]: Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli, "ChatGPT Outperforms Crowd Workers for Text-Annotation Tasks," *Proceedings of the National Academy of Sciences* 120, no. 30 (2023): e2305016120, https://doi.org/10.1073/pnas.2305016120.
[^25]: Douglas Biber and Edward Finegan, "Adverbial Stance Types in English," *Discourse Processes* 11, no. 1 (1988): 1-34, https://doi.org/10.1080/01638538809544689.
