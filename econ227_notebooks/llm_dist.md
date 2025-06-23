---
author:
- Yash Mali
authors:
- Yash Mali
html: true
title: LLM Probability Distribution
toc-title: Table of contents
---

## What do Large Language Models do?

Large language models like ChatGPT do something that seems very simple:
**Next word prediction.**

What does that mean? It means that given a sequence of words, the model
predicts the next word in the sequence. For example, if the input is
"The cat sat on the", the model might predict "mat" as the next word.

:::::: {.cell layout-align="default"}
::::: cell-output-display
<div>

`<figure class=''>`{=html}

<div>

![](llm_dist_files/figure-markdown/mermaid-figure-1.png){width="7.99in"
height="0.73in"}

</div>

`</figure>`{=html}

</div>
:::::
::::::

We saw an example of predicting just one word. These models predict only
one word at a time, but they can do this for very long sequences of
words.

For example: "bob went to the store" to buy some milk.

:::::: {.cell layout-align="default"}
::::: cell-output-display
<div>

`<figure class=''>`{=html}

<div>

![](llm_dist_files/figure-markdown/mermaid-figure-2.png){width="12.18in"
height="6.9in"}

</div>

`</figure>`{=html}

</div>
:::::
::::::

What the model is doing is learning the probability distribution of the
next word given the previous words.

The probability of predicting the next word $w_{t}$ given the previous
words $w_1, w_2, \ldots, w_{t-1}$ is:

$$
P(w_t \mid w_1, w_2, \ldots, w_{t-1}) = \frac{P(w_1, w_2, \ldots, w_{t-1}, w_t)}{P(w_1, w_2, \ldots, w_{t-1})}
$$

LLMs approximate this probability:

$$
P(w_t \mid w_1, w_2, \ldots, w_{t-1})
$$

The model predicts the next word by selecting the word $w_t$ that
maximizes this conditional probability:

$$
\hat{w}_t = \arg\max_{w} P(w \mid w_1, w_2, \ldots, w_{t-1})
$$

These models give a probability distribution over the entire vocabulary
(all the words the model was trained on). We can then pick the word with
the highest probability as the next word or we can sample from this
distribution to get more varied (creative) outputs.

:::: {.cell execution_count="1"}
::: {.cell-output .cell-output-display}
![](llm_dist_files/figure-markdown/cell-2-output-1.png)
:::
::::

Note that while this is with words and language, the same idea applies
to any sequential data, like stock prices, weather data, etc. The model
learns the probability distribution of the next value given the previous
values.

:::: {.cell execution_count="2"}
::: {.cell-output .cell-output-display}
![](llm_dist_files/figure-markdown/cell-3-output-1.png)
:::
::::

Typically, you would use more inputs to the model than the price
history. This can public sentiment, news, or other features that might
affect the price. The model learns the joint distribution of these
features and the price, allowing it to make predictions about future
prices.
