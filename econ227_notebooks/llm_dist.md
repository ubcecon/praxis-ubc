---
author:
- Yash Mali, Kaiyan Zhang
authors:
- Yash Mali, Kaiyan Zhang
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

![](llm_dist_files/figure-markdown/mermaid-figure-2.png){width="12.09in"
height="5.08in"}

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

Lets look at an example of how this works in practice:

:::: {.cell execution_count="1"}
::: {.cell-output .cell-output-display}
![](llm_dist_files/figure-markdown/cell-2-output-1.png)
:::
::::

To get more creative responses you change the distribution at the output
where you pick the next word. Very simply this involves making the
distribution sharper or flatter. If you make the distribution sharper,
you are more likely to pick the word with the highest probability. If
you make it flatter, you are more likely to pick a word that is not the
most probable one.

This is called **temperature**. A higher temperature makes the
distribution flatter, while a lower temperature makes it sharper. You
would want to use a temperature of more than 1 $(1.2-1.5)$ for creative
responses, and a temperature of less than 1 $(0.1 - 0.5)$ for more
focused responses. For a balanced response, you can use a temperature of
$0.7-1$. Another set of parameters are called top-p and top-k sampling.

:::: {.cell execution_count="2"}
::: {.cell-output .cell-output-display}
![](llm_dist_files/figure-markdown/cell-3-output-1.png)
:::
::::

In the example above, we see how the probability distribution changes
with different temperatures. A high temperature (1.5) results in a
flatter distribution, meaning the model is more likely to sample from
less probable tokens, while a low temperature (0.5) results in a sharper
distribution, favoring the most probable tokens.

Note that while this is with words and language, the same idea applies
to any sequential data, like stock prices, weather data, etc. The model
learns the probability distribution of the next value given the previous
values.

:::: {.cell execution_count="3"}
::: {.cell-output .cell-output-display}
![](llm_dist_files/figure-markdown/cell-4-output-1.png)
:::
::::

Typically, you would use more inputs to the model than the price
history. This can public sentiment, news, or other features that might
affect the price. The model learns the joint distribution of these
features and the price, allowing it to make predictions about future
prices.
