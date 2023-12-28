# Transformers

- watch https://www.youtube.com/watch?v=zxQyTK8quyY
- watch https://youtu.be/SZorAJ4I-sA and then https://youtu.be/TQQlZhbC5ps
- positional encoding, attention, self-attention, multi-head attention

1. encode positions in the data, not in the neural network structure itself
2. attention <-- e.g.: look at entire sentence and learn from data which words to pay attention to when translating into a target language word
3. self-attention = attention but also context of other words in the input sentence to help disambiguate the polysemic meaning of a word

- embedding = meaning
- positional encoding = context ([because just using embeddings isn't enough: think "bank robber" vs "river bank"](https://www.cloudskillsboost.google/course_sessions/6505024/video/363235))
- multi-head attention = multiple weighted contributions of context
- feed-forward / linear = digest
- softmax = probabilities of possibilities of next word in the target output language for example
