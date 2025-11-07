# Lesson Plan: Zero-Shot Natural Language Inference (NLI) on Historical Texts

*Praxis UBC Team<br>Kaiyan Zhang, Irene Berezin, Alex Ronczewski, Krishaant Pathmanathan*

November, 2025

## Objective

This lesson is aiming to demonstrate how to use zero-shot natural language inference (NLI) models to assist in analyzing and interpreting historical texts. We consider NLI as a powerful tool for historians to draw inferences from historical documents, enabling them to uncover relationships, contradictions, and implications within the texts without requiring extensive labeled datasets for training. 

We will conduct a case study using a small corpus of legal documents about cases involving early Chinese immigrants in British Columbia, Canada, to illustrate how NLI can help historians draw inferences and gain insights from historical data. This case study will also highlight the challenges and considerations when applying NLI models to historical texts, such as dealing with archaic language, OCR errors, and contextual understanding.

In general, this lesson will fill in a gap in the existing resources on Programming Historian by providing a practical guide on using NLI for historical text analysis, which skills may also be applicable to other domains of digital humanities research. We therefore hope this lesson will enrich the resources available to historians and digital humanists interested in leveraging NLP techniques for their research, and promote the adoption of advanced AI tools in the field of history.

## Dataset/Corpus

The dataset consists of rulings, acts, and commission reports from Canadian legal archives, specifically focusing on cases involving early Chinese immigrants in British Columbia. The documents are in English and date back to the late 19th and early 20th centuries. 

The central document for our case study is the "Chinese Regulation Act, 1884" and the related legal cases "Regina v. Wing Chong (1885)" and "Regina v. Mee Wah (1885)". Additional documents include commission reports and rulings that provide context and details about the legal environment of the time. 

The PDF versions of these documents can be found in the Canadian Legal Information Institute (CanLII) archives, and we have pre-processed them into plain text for analysis. 

## Tools and Libraries

We will use Python 3.12 as our programming language and Jupyter as our primary development environment. Supporting Python scripts may also be used to carry pre-defined functions or classes that are too cumbersome to include directly in the lesson.

The following libraries will be utilized:
- `transformers` from Hugging Face for accessing pre-trained NLI models.
- `torch` for tensor computations and model inference.
- `numpy` for numerical operations.
- `scikit-learn` for additional machine learning utilities.
- `pandas` for data manipulation and organization.
- `nltk` for text processing and tokenization.
- `matplotlib` and `seaborn` for data visualization.

## Model Choice

For the purpose of this lesson, we will have to use a pre-trained zero-shot NLI model, for which we will use the models available in the Hugging Face that are: 1) Open-source, 2) fine-tuned for NLI tasks, and 3) capable of handling English text, preferably also legal English text, effectively. 

The justification for using a zero-shot model is that it allows us to perform inference without the need for task-specific training data, which is particularly useful when dealing with historical texts where labeled data may be scarce or non-existent. This approach enables historians to leverage advanced NLP techniques without extensive computational resources or expertise in model training, it is also cost-effective in terms of time and resources, if only unlabeled data is available.

## Lesson Structure

1. **Introduction**: We will begin by briefly introducing the concept of Natural Language Inference (NLI) and its relevance to historical text analysis. We will discuss the challenges historians face when interpreting historical documents and how NLI can assist in drawing inferences from such texts. We will also provide a background overview of our case study and why we see it as a suitable example for applying NLI techniques.

2. **Suggested Prior Skills**: In this section, we will outline the necessary skills and knowledge that students should have before starting the lesson. This includes basic Python programming, familiarity with Jupyter notebooks, some basic knowledge of NLP, and computational text analysis. We will also provide links to resources for students to acquire these skills if they do not already possess them.

3. **Lesson Setup**: In this section, we will introduce two ways to set up the lesson environment: using Google Colab for a cloud-based solution and setting up a local environment with Anaconda. We will provide step-by-step instructions for both methods, including installing necessary libraries and configuring the environment.

4. **Data Preparation**: This section will introduce the process of loading and preparing the historical text data for analysis, including a short subsection on the OCR process we used to convert scanned documents into machine-readable text. We will cover text cleaning, tokenization, and formatting the data for input into the NLI model.

5. **Fundamentals of NLI**: Here, we will explain the core concepts of Natural Language Inference in detail, including definitions of entailment, contradiction, and neutrality. We will discuss how NLI models work, the architecture of transformer-based models, and the significance of zero-shot learning in this context.

6. **Zero-Shot NLI with Pre-trained Models**: In this section, we will discuss how to use pre-trained zero-shot NLI models to perform inference on our corpus of historical texts. We will also cover how to choose appropriate models for the specific task of analyzing historical documents, and how to access these models from various libraries and platforms.

7. **Prompt Engineering for NLI**: This section will focus on the importance of prompt engineering in zero-shot NLI tasks. We will provide guidelines and best practices for crafting effective labels and hypotheses that can help improve the performance of NLI models when analyzing historical texts. We will also include examples of well-constructed prompts relevant to our case study.

8. **Interpretation of Results**: Here, we will provide guidance on how to interpret the outputs of the NLI model, including understanding confidence scores and the implications of different inference results. We will discuss how to critically evaluate the model's predictions in the context of our case study.

9. **Conclusion and Further Resources**: In the final section, we will summarize the key takeaways from the case study and provide additional resources for students who wish to explore NLI and historical text analysis further. This may include links to research papers, tutorials, and datasets for further practice. A glossary of key terms used throughout the lesson will also be provided for reference.
