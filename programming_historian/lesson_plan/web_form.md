## Lesson Proposal for Programming Historian (English edition)

_Programming Historian_ in English is inviting proposals for new lessons.

We encourage prospective authors to think carefully about how their proposal could enhance our learning offer. You can [explore our journal](https://programminghistorian.org/en/lessons/) to discover what’s already available and consider what you might be able to add.

If the method or approach you’re interested in writing a lesson about is already represented by the [Spanish](https://programminghistorian.org/es/lecciones/), [French](https://programminghistorian.org/fr/lecons/), or [Portuguese](https://programminghistorian.org/pt/licoes/) editions of _Programming Historian_, we welcome proposals to translate those existing, original lessons into English. In this call, we would particularly like to encourage proposals for translations. Please review the list of translations we are prioritising for development. 

Questions? Please write to the Managing Editor of _Programming Historian in English_, Alex Wermer-Colan (english@programminghistorian.org) or our Publishing Manager, Anisa Hawes (admin@programminghistorian.org).

---

Name:  
Email:

What kind of lesson are you proposing?
- [ ] An original English-language lesson *Please go to Section 2*
- [ ] A translation into English from an existing, original Spanish, French, or Portuguese lesson *Please go to Section 3*

---

### Section 2: Original English-language lessons

Answer the following questions if you are proposing an *original* English-language lesson.

a). What is your proposed lesson title?

Natural Language Inference for Historical Text Analysis Using Python

We suggest a short, descriptive title:

- Begin with verb or a noun to define the main learning activity, method or process.
- Identify the kind of data readers will handle in the lesson.
- Name key tools, software libraries or programming languages readers will use.


1. What can readers expect to learn from your proposed lesson?
[3-4 sentences]

This lesson is aiming at teaching readers the basics of natural language inference and how to apply NLI techniques to analyze historical texts in Python. We propose a lesson with two parts: 1) an introduction to the fundamentals of NLI and the traditional methods of inference in NLP, and 2) zero-shot NLI using pre-trained models. We will illustrate the application of NLI techniques through a case study analyzing a corpus of legal documents related to early Chinese immigrants in British Columbia, Canada. By the end of the lesson, readers will be able to understand NLI concepts, set up a Python environment (both locally and online through Google Colab) for NLI tasks, and apply zero-shot NLI models to draw inferences from historical texts.

2. Please tell us how your proposal could support, expand, or supplement the lessons we've already published.
Use these questions to structure your answer:
- To which existing _Programming Historian_ lesson(s) could your proposal provide a foundational introduction?
- To which existing _Programming Historian_ lesson(s) could your proposal provide an advancement or extension?
- How could your proposal fill a gap in our lesson directory?  
[200-300 words]

Currently, Programming Historian has a variety of lessons focusing on text analysis, word embeddings and topic modeling, but there is a noticeable gap in resources specifically addressing Natural Language Inference (NLI) and its applications in historical text analysis. While some lessons touch on NLP techniques, none provide a comprehensive introduction to NLI, particularly zero-shot NLI, which allows for inference without the need for labeled training data, which we found particularly useful for historical texts where labeled data is often scarce.

We see this lesson as an extension to the lesson *Understanding and Creating Word Embeddings*, which introduced readers to word embeddings and their applications in historical text analysis. Our proposed lesson would build upon that foundation by introducing other approaches to understanding text embeddings, such as pooling methods for stance inference, and then advancing into NLI through pre-trained models.

In addition, the case study we propose, focusing on legal documents will fill a gap in the existing resources that rarely address the specific challenges of analyzing historical legal texts, such as archaic language and OCR errors. By providing practical guidance on using NLI models in this context, we aim to equip historians and digital humanists with new tools and methodologies for their research.

3. Please share some insights into how you came to use this method or tool as part of your work within the humanities:  
[100-300 words]

In my work, I often encounter historical texts that require deep analysis to uncover implicit meanings, relationships, and contradictions. Traditional NLP techniques like topic modeling and word embeddings have been useful, but I found them limited when it comes to drawing nuanced inferences from texts, especially when seeking to understand stances or implications that are not explicitly stated.

However, labeling data for supervised learning tasks in historical contexts is often impractical due to the scarcity of annotated datasets. This led me to explore zero-shot learning approaches, particularly Natural Language Inference (NLI) models, which can infer relationships between text pairs without needing task-specific training data. While challenge still exists in finding effective models and applying these models to historical texts due to language evolution and OCR errors, I found that with careful prompt engineering and data preparation, NLI models can yield valuable insights. 

Therefore, I believe that sharing this method through a Programming Historian lesson could greatly benefit other researchers in the humanities who face similar challenges in analyzing historical texts.

4. Please tell us about the research case study you propose centring within your lesson.
- Successful lessons centre real datasets and sample code that readers can handle and experiment with.
[100-300 words]

The research case study we propose centers on a corpus of legal documents related to early Chinese immigrants in British Columbia, Canada, during the late 19th and early 20th centuries. This corpus includes rulings, acts, and commission reports, with a particular focus on the "Chinese Regulation Act, 1884" and related legal cases such as "Regina v. Wing Chong (1885)" and "Regina v. Mee Wah (1885)". These documents provide a rich context for exploring how NLI can be applied to historical texts, as they contain complex legal language and implicit relationships that are not always explicitly stated.

Our study specifically focus on the two judges' rulings in the cases mentioned above, analyzing how NLI models can help infer stances, contradictions, and entailments within the legal arguments presented. We will demonstrate how to prepare the text data, using embedding methods to mine for stance inference and topic alignment, and then apply zero-shot NLI models using Python, and interpret the results in the context of historical legal analysis. 

In our lesson, we will provide readers with access to the pre-processed text data and sample code that they can use to replicate the analysis. This hands-on approach will allow readers to experiment with NLI techniques and gain practical experience in applying these methods to historical texts. We would also introduce interactive visualizations created using Plotly Python library to help readers better explore and understand the inference results.

5. Please outline how your choice of software and data would support our commitment to openness:
We advocate for the use of open source software, open programming languages, open access datasets wherever possible.
Use these questions to structure your answer:
- Which open source software, open tools, open programming languages, or open datasets does this lesson make use of?
- Which (if any) proprietary software or commercial tools does this lesson make use of? We strongly recommend authors choose open source alternatives.
- What (if any) costs are required to use this tool? Does access require users to supply credit card information?
[100-300 words]

The lesson will primarily utilize open source software and tools to ensure accessibility and support our commitment to openness. We will use Python, an open-source programming language widely adopted in the digital humanities community, along with Jupyter Notebooks for an interactive coding environment.

For NLP tasks, we will leverage the Hugging Face Transformers library, which is open source and provides access to a variety of pre-trained models suitable for Natural Language Inference (NLI). Additionally, we will use other open-source libraries such as NumPy, Pandas, NLTK, Spacy NLP and Plotly for data manipulation, text processing, and visualization.

The dataset we will use for the case study consists of legal documents that are publicly available through the Canadian Legal Information Institute (CanLII) archives. We will provide pre-processed versions of these texts in plain text format, ensuring that readers can easily access and work with the data without any restrictions.

The models and libraries we will use do not require any proprietary software or commercial tools, making them freely accessible to all users. There are no costs associated with using the tools and datasets presented in this lesson, and users will not need to supply credit card information to access any of the resources. To access more hugging face models, users may need to create a free account on the Hugging Face website, but this does not involve any payment information for basic access.

To handle the possible computational limitations of some users, we will also provide instructions for using Google Colab, a free cloud-based platform that allows users to run Python code without needing to install software locally. Google Colab does not require users to supply credit card information for basic access, making it an accessible option for most users.

6. Please provide us with some information about how your method or tool could be applied or adapted for use in languages other than English:
- We have a strong preference for methodologies and tools that can be used in multilingual research-contexts.  
[100-300 words]

Since the release of BERT and other transformer-based models, there has been significant progress in developing multilingual NLP models that can handle a variety of languages. Many pre-trained models available through the Hugging Face Transformers library, such as bert-base-multilingual-cased, are designed to work with multiple languages, making them suitable for NLI tasks in non-English contexts.

In our lesson, while we will focus on English-language historical texts, the methodologies we present can be adapted for use with texts in other languages. We will also leave out a section discussing where and how to find suitable multilingual NLI models, as well as considerations for handling language-specific challenges, such as tokenization and cultural context.

Furthermore, the data preparation techniques and prompt engineering strategies we discuss can be generalized to other languages, allowing researchers to apply NLI techniques to historical texts in their native languages. We will encourage readers to explore multilingual datasets and experiment with different models to see how well they perform on texts in various languages.

7. Please outline any technical prerequisites and potential limitations of access to using this method or tool:
- Our readers work with different operating systems and have varying computational resources.  
[100-200 words]

The technical prerequisites for this lesson include a basic understanding of Python programming and familiarity with Jupyter Notebooks. Readers should also have some foundational knowledge of NLP concepts, although we will provide introductory explanations of key terms related to NLI. We would recommend readers to first complete introductory lessons on Python, Jupyter, text processing, and word embeddings available on Programming Historian if they are new to these topics.

To accommodate readers with varying computational resources, we will provide two options for setting up the lesson environment: a local setup using Anaconda and a cloud-based solution using Google Colab. The local setup may require more computational power, especially when working with large models, while Google Colab offers free access to GPUs, making it more accessible for users with limited hardware capabilities.

8. Optional link to sample code or a draft extract of this proposed lesson on your personal GitHub repository:

The case study is based on a workshop presented at APSA pre-conference in 2025. You can find the relevant information here: <https://migration.ubc.ca/news/apsa-pre-conference-narrative-and-text-analysis-in-the-study-of-migration-and-citizenship/>.

A rendered html version of the workshop notebook can be found here: <https://ubcecon.github.io/praxis-ubc/docs/hist_workshop/text_embeddings_workshop.html>.

---

### Section 3: Translations into English

Answer the following questions if you are proposing a translation into English from an existing, original Spanish, French, or Portuguese lesson.

a). What is the title of the lesson you want to translate (in the source language)?

b). What is your proposed translation of the title into English?

c). Please share a link to the lesson you want to translate:


1. Please provide a 300-400 word translation sample. We suggest translating the opening paragraphs of the lesson you'd like to translate. 
[300-400 words]

2. Why do you think this particular lesson would be valuable to translate into English?
- Please review the list of translations we are prioritising for development in our blog post. 
- If the lesson you propose translating isn't on this list, we'd still like to hear from you. 
[200-300 words]

3. Please share some insights into how you came to use this method or tool as part of your work within the humanities:
[100-300 words]

4. Please tell us how you plan to adapt or localise the research case study centred by the original lesson for an English-language readership.
- Successful lessons centre real datasets and sample code that readers can handle and experiment with.
[100-300 words]

5. Optional link to sample code or a draft extract of this proposed translation on your personal GitHub repository:


---

Please send this form to our Publishing Manager, Anisa Hawes (admin@programminghistorian.org).