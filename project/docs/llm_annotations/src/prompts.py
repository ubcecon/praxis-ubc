"""The annotation prompt. Its text is part of the cache key, so editing it means
there are no saved answers to read back."""

RUBRIC = """You are annotating reader comments from a Canadian news website.

A comment is CONSTRUCTIVE if it tries to add something to the conversation: it \
makes a specific point, gives evidence or a personal experience, offers a \
solution, or engages with the article's argument.

A comment is NOT CONSTRUCTIVE if it is only an insult, a one-line dismissal, \
sarcasm with no substance, off-topic ranting, or an unsupported assertion.

Comment:
\"\"\"{comment}\"\"\"

Reply in exactly this format and nothing else:
LABEL: yes
REASON: <one short sentence>"""
