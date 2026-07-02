# Reviewer Feedback and Responses

Feedback from Laura on the NLI lesson. Each point shows Laura's feedback in bold, then a short list of what we did so far. This is the first pass, we can keep adding to it.

---

## General Comments

**Clarifying the central workflow and the role of each method. I'd recommend streamlining the lesson around the specific aim of teaching how to use zero-shot Natural Language Inference to analyze stance in historical texts. Currently, the key method and real heart of the lesson comes more than halfway through and the lesson seems quite methodologically ambitious, covering TF-IDF, lexicon analysis, fuzzy matching, embeddings, UMAP, topic anchors, zero-shot classification, validation, label sensitivity, quote sensitivity, and bootstrap confidence intervals. These are all potentially useful, but the relationship between the different stages is currently not always clear. I'd recommend making clear from the start that zero-shot NLI for stance analysis is the main method of the lesson and tightening the focus around that central NLI-based workflow. TF-IDF, lexicon analysis, embeddings, and UMAP could be more clearly folded into this main NLI workflow as supporting context, baselines, or optional exploratory steps (or even removed if they are not essential). For each major method or tool, it would be helpful to explain: what it is, why it is useful, how it is being used, and how it connects to the previous and next stages.**

What we did:

- Made zero-shot NLI the main method. Rewrote the abstract and the opening paragraph around it, and renamed the core section to "Stance Classification with Zero-Shot NLI".
- Cut the methods that pushed the main point past the halfway mark: TF-IDF, the lexicon baseline, embeddings and Sentence-BERT, UMAP, topic anchors, and the topic vs stance correlation section.
- Kept the one useful idea from the lexicon baseline as a short "why not just count keywords?" paragraph at the start of "Why NLI?".
- Each remaining step now says what it is, why it is there, and how it feeds the next step.

**Restructuring and adding a clear roadmap early in the lesson. It would be helpful to provide readers an overview of the full workflow near the beginning, and then add signposting throughout. The workflow and table of contents could be restructured around clearer "milestone" stages (e.g., context/question > preparation > NLI analysis > evaluation > interpretation/adaptation).**

What we did:

- Added a four stage roadmap near the top: data preparation, stance classification, evaluation and robustness, interpretation.
- Rebuilt the section order and table of contents around those stages.
- Quotation removal and passage selection are now subsections under data preparation.
- Added a "you are in stage X of 4" line at the start of each of the four stage sections, so readers can see where they are in the workflow.

**Strengthening the pedagogical scaffolding throughout. The lesson would benefit from more explanation, examples, and checks. Break longer code blocks into smaller chunks, explain what each step does in the main body, and show intermediate outputs. There are places where the lesson interprets results without showing the output. For example, after TF-IDF the lesson discusses the results but the reader never sees the TF-IDF output.**

What we did:

- Show the output for every step the text interprets: group counts, the sentence and window score tables, the evaluation metrics table, the quote sensitivity table, and the label sensitivity table.
- Split the quotation removal code into smaller steps with a line of explanation each.
- Added a short "small label design check" section that explains what makes a good label set.
- The TF-IDF example is fixed by removing TF-IDF, and the steps we kept now show the tables the text talks about.

**Making the lesson page the primary teaching resource and including the full workflow. Please ensure the full sequence of steps with code is in the lesson page. The draft refers to a notebook several times, but the lesson page is the primary resource and should contain everything a reader needs.**

What we did:

- Removed every "the notebook demo loads" line.
- Each heavy step now shows its full code in the lesson, plus an in page option to load the saved CSV instead.
- Expanded the quotation removal steps so the whole process is on the page, not in a separate notebook.

**Addressing computational accessibility and clarifying the role of precomputed results. The NLI steps are slow (60 to 90 minutes on CPU). Provide a lower resource route as well as the full workflow, and clearly separate the two.**

What we did:

- Added a lower resource path to every slow step: load a saved CSV instead of running the model, with the expected numbers shown so readers can still follow along.
- "Setting Up the Pipeline" now states the two options plainly: load the saved CSVs, or run the scoring cells to redo it from scratch.
- Added a run_notes dictionary that records the model, template, and mode used for the saved outputs.

Still to do: the Prerequisites section could also summarize the two paths.

**Clarifying multilingual adaptation. This case study uses English models and English texts. Acknowledge this and point to resources for other languages, including multilingual or language specific models.**

What we did:

- Added a "Working with Other Languages" section: use a sentence splitter for the target language, pick a multilingual or language specific NLI model (XLM-R, mDeBERTa), write the labels and template in the corpus language, and test on known examples first.
- Noted that the English resources (Historical Thesaurus, COHA) will not help for other languages, and to find period and language specific references instead.

Still to do: could add a short note in Software and Setup on which parts are English only (the spaCy model).

**Adding more practical guidance on adapting the workflow to other corpora. When discussing model choice or label design, give more guidance on how readers can make these decisions for their own corpus (where to find a model's training data and intended use, how to test labels on known examples, how to spot historical vocabulary issues, which steps they would need to recompute).**

What we did:

- Added an "Adapting to Your Own Corpus" section with the five decisions a reader has to make (corpus, keywords, labels, model, evaluation sample).
- "Choosing a Model" now covers checking the training data, recording the exact model and settings, and testing on known examples.
- Added the label design check for testing labels on passages you already know.
- Quotation removal now covers how the threshold was chosen and other options for bigger or noisier corpora.
- Kept "Digital Resources for Historical Semantics" for checking historical vocabulary.

---

## Section by Section

### Lesson Goals

**I'd suggest expanding the introductory paragraph into a fuller "Introduction" section, with "Lesson Goals" as a subsection. This could define key concepts (NLI, zero-shot classification, what "stance" means, labels and hypotheses), explain why the approach is useful, and give an overview of the workflow.**

What we did:

- Rewrote the opening paragraph to explain NLI, zero-shot, and hypotheses in plainer terms, and added the four stage roadmap.

Still to do: split this into an "Introduction" section with "Lesson Goals" under it, and move the concept definitions (stance, labels vs hypotheses) up there.

**The lesson goals may need to be revised to match the new workflow. Right now TF-IDF, lexicon, embeddings, UMAP, classification, validation, and robustness all read as equally central. The goals should show which steps are central and which are supporting.**

What we did:

- Rewrote the goals list. Dropped the TF-IDF, embeddings, and UMAP goals. It now covers quotation removal, passage selection, zero-shot NLI, label design, evaluation, and robustness.
- Changed "labeled ground truth set" to "manually labelled evaluation sample".

### Prerequisites

**More clarity on computational access and background knowledge. Give guidance on the compute requirements and the different paths through the lesson, and separate the lower resource path from recomputing everything from scratch.**

What we did:

- The two paths are now built into the body of the lesson.

Still to do: state the two paths in the Prerequisites section too.

**Align the prerequisites with the difficulty. The section says "intermediate Python experience," but the lesson uses transformers, NLI, similarity measures, validation metrics, and bootstrap intervals. Set clearer expectations about what readers should already know and what the lesson will introduce.**

What we did:

- Removed the word embeddings prerequisite link, since embeddings were cut.

Still to do: spell out the assumed background versus what the lesson teaches.

### Software and Setup

**Explain what the main packages are for and how they fit the workflow (data handling, sentence splitting, NLI, visualization), and note any multilingual features (for example the spaCy model is English specific).**

What we did:

- Added a line saying what each package is for: pandas and numpy for tables, nltk and spacy for sentence splitting, transformers and torch for the NLI model, matplotlib and seaborn for figures.
- Trimmed the install list to what the lesson now uses (removed umap-learn, scikit-learn, sentence-transformers), and removed Sentence-BERT from Software Versions.

Still to do: note which parts are English only inline (currently in "Working with Other Languages").

**Say where to run each command. The `python -m spacy download en_core_web_sm` block is a terminal command, not Python.**

What we did:

- Split the setup into "Download NLTK data inside Python" and "Then download the English spaCy model from the terminal", so the terminal command is labelled.

### Downloading the Data, Case Study, Preparing the Corpus

**Introduce the case study, corpus, and research question before asking readers to download the files. Add more on how the files were made, how they are used, and what readers should keep in mind for their own materials.**

What we did:

- Added a paragraph describing the source documents, that the text files came from OCR, and how they are used.

Still to do: the download section still comes before the case study, so consider moving the case study first.

**Include provenance and reuse status: where the scans came from, whether they are public, who did the OCR, and whether the data is reusable.**

What we did:

- Added a provenance note covering these questions.
- Wrote it as a note to ourselves for now: before publication, confirm the scan sources, who did the OCR, and the reuse status.
- Added a note that the current data folder is for review and the public download link needs checking before publication.

Needs authors: fill in the real provenance and reuse answers.

**Describe what the code does, even when simple. In "Preparing the Corpus" the code loads the metadata and makes a group column (Crease, Begbie, Regulation Act, Other). Say this in the text and show how to check the group counts.**

What we did:

- Added text explaining the grouping, and added df["group"].value_counts() with its expected output.

### Detecting and Removing Direct Quotations

**Show more of the process. The current code does not show the full quotation removal workflow. Expand it so readers see the whole thing: sentence split Crease and the Act, compute similarity, inspect examples above and below the threshold, remove or flag matches, explain how the threshold was chosen, save or load the cleaned text, show before and after, and say which version is used later.**

What we did:

- Expanded it into a full workflow: split Crease and the Act into sentences, build a quote_scores table, look at the top matches, then apply the 0.6 threshold to get the cleaned sentences.
- Added a "Checking the Threshold" section that looks at the sentences just above and below the cutoff and explains how to read them.
- Explained how the threshold was chosen, the limits of SequenceMatcher, and other options for bigger or noisier corpora.
- Said the cleaned texts are saved to data/texts/quotations_removed/ and that the NLI analysis uses those versions.

Still to do: could add an actual before and after sentence pair, not just the counts.

### Validation and Robustness Checks

**These sections matter because they tell readers how much to trust the outputs. Explain how the 45 labelled snippets were made: who labelled them, was there more than one annotator, was there a guide, how were ambiguous cases handled, why this sample size. Consider "manually labelled evaluation sample" instead of "ground truth", since these are interpretive categories.**

What we did:

- Replaced "ground truth" with "manually labelled evaluation sample" everywhere (heading, abstract, goals), and added a line that these are interpretive categories, not fixed reference labels.
- Added an error check that loads and reads the wrong predictions, and framed the accuracy as support for close reading, not a replacement for it.
- Added a note on the bootstrap section that the intervals only cover sampling variation, not OCR, label, or model uncertainty.

Needs authors: fill in how the 45 snippets were labelled (who, how many annotators, guide, ambiguous cases, sample size).

---

## Still Open

- Data provenance: confirm scan sources, who did the OCR, whether the scans are public, and the reuse status.
- Annotation notes: document who labelled the 45 snippets and how.
- Data link: check and update the public download link before publication.
- Introduction: consider a full "Introduction" section with Lesson Goals under it.
- Prerequisites: add the two path summary and the assumed background.
- Section order: consider moving the case study before the download step.
- English only note: flag the spaCy model and other English parts in Software and Setup.
- Before and after example in the quotation removal section.
