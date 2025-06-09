## Transformers : ...

Author: *Krishaant Pathmanathan, PRAXIS UBC Team*

Date: 2025-06


```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import plotly.graph_objects as go
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


```

## Prediction Game: Where Will the Ball Go?

Transformers are all about **making predictions from context**. But what does that really mean? Before diving into tokens and attention, let’s play a game.

**Imagine this:**
You see a bouncing ball mid-air. Where will it go next? How do you know? Just like us, models try to guess what comes next based on what they've already seen.


## Given an image of a ball can you predict where it will go next ?
<div style="display: flex; gap: 20px;">
  <img src="data/static_ball2.png" alt="Static ball" width="300"/>
  <img src="data/moving_ball3.png" alt="Moving ball" width="300"/>
</div>

Given a sequence, can you now tell ?


Turns out there are many sequences in the world - words in a sentence, frames in a video, notes in a melody, steps in a recipe. The challenge isn't just seeing them, it's predicting what comes next. To understand or predict them, we need a model that doesn’t just look at things in isolation…  It has to **remember what came before**.

That’s where **Recurrent Neural Networks (RNNs)** come in.

<!-- maybe insert some pictures of that here  -->


## Sequential Prediction with RNNs

Imagine reading one word at a time, keeping track of what came before... that's what an RNN does. RNN stands for **Recurrent Neural Network**. It’s a type of model that learns from **sequences** — like sentences, music, or even time. It's an important precursor to a **transformer**

### How It Works (Step-by-Step)

1. **Give it a sentence** — for example:  
   `"I love recurrent neural ____"`

2. **Turn words into numbers** (this is called *tokenizing*)  
   > Computers can’t understand words — they only understand numbers!

3. **Send the numbers into the RNN** — one by one

4. **At each step**, the RNN tries to **remember what came before**  
   > It passes a little memory called a **hidden state** from word to word

5. **At the end**, it **guesses what comes next!**  
   > Like filling in the blank at the end of the sentence



```python
# Simulated RNN function (takes a look at past to make prediction)
def my_rnn(word, hidden_state):
    # Imagine the RNN does something with the word and memory
    new_hidden_state = [h + 1 for h in hidden_state]  # update memory
    prediction = "networks!" if word == "neural" else None
    return prediction, new_hidden_state

# Initial hidden state
hidden_state = [0, 0, 0, 0]

# Input sentence (tokenized)
sentence = ["I", "love", "recurrent", "neural"]

# RNN step-by-step loop
for word in sentence:
    prediction, hidden_state = my_rnn(word, hidden_state)

# Final predicted word
next_word_prediction = prediction
print("Next word prediction:", next_word_prediction)

```

    Next word prediction: networks!


###  Why RNNs Can Be Tricky ?

- They **read one word at a time**, so it’s slow
- They **forget things** after a while (just like people!)
- They **can’t look at everything at once**


RNNs are like someone reading a story **out loud, one word at a time**. Transformers (like GPT) are like someone **looking at the whole page at once**.

**RNNs walked, so Transformers could fly.**

## What is a GPT model ?

A GPT is a Generative Pre-Trained Transformer. The first two words are self-explanatory: generative means the model generates new text; pre-trained means the model was trained on large amounts of data. 

What we will focus on is the **transformer** aspect of the language model, the main proponent of the recent boom in AI.

## What's a transformer ?

A transformer is a neural network that learns context and thus meaning by tracking relationships in sequential data like the words in this sentence.
It is the main component that underlies tools like ChatGPT. It can trained to take in a piece of text, maybe even with some surrounding images or sound accompanying it, then produce a prediction of what comes next, in the form of a probability distribution over all chunks of text that might follow.

*Note there are many other types of transformers (voice-to-text, text-to-image, etc.). 

<img src="data/transformerpredict.png" alt="Transformer Prediction" width="800"/>


## Transformer Applications 

To further understand how transformers work we will walk through examples of these transformers being used. For now you can think of transformers as a black box, but there are componenets within it that allow it to understand context, exceeding the capabilities of an RNN. 



## Tokenization

Just like text is broken into tokens, images are broken into patches, and audio is split into time steps or spectrogram slices before being passed to transformer models like ViT or Wav2Vec2.


```python
from transformers import AutoTokenizer

# Load a BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize an example text
text = 'The ball is round.'
tokens = tokenizer(text)
tokens
```




    {'input_ids': [101, 1996, 3608, 2003, 2461, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}



## Word Prediction Demo


```python
from transformers import pipeline

generator = pipeline('text-generation')
prompt = 'The history of the world is'
output = generator(prompt, max_length=20)
print(output[0]['generated_text'])
```

    No model was supplied, defaulted to openai-community/gpt2 and revision 607a30d (https://huggingface.co/openai-community/gpt2).
    Using a pipeline without specifying a model name and revision in production is not recommended.



    config.json:   0%|          | 0.00/665 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/548M [00:00<?, ?B/s]



    generation_config.json:   0%|          | 0.00/124 [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/26.0 [00:00<?, ?B/s]



    vocab.json:   0%|          | 0.00/1.04M [00:00<?, ?B/s]



    merges.txt:   0%|          | 0.00/456k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/1.36M [00:00<?, ?B/s]


    Device set to use cpu
    Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.


    The history of the world is inextricably linked with its history in Europe," said Gutt


## The Attention Block

The Attention Block where they communicate with each other to update their values based on context. For example, the meaning of the word model in the phrase a machine learning model is different from its meaning in the phrase a fashion model. The Attention Block is responsible for figuring out which words in the context are relevant to updating the meanings of other words and how exactly those meanings should be updated


```python
import numpy as np
from scipy.special import softmax

# Toy example of query-key-value
Q = np.array([[1, 0]])
K = np.array([[1, 0], [0, 1]])
V = np.array([[10], [20]])

attention_scores = Q @ K.T
attention_weights = softmax(attention_scores, axis=1)
output = attention_weights @ V
output
```




    array([[12.68941421]])






