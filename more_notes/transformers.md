# Transformers

- watch https://www.youtube.com/watch?v=zxQyTK8quyY
- watch https://youtu.be/SZorAJ4I-sA and then https://youtu.be/TQQlZhbC5ps
- positional encoding, attention, self-attention, multi-head attention

1. encode positions in the data, not in the neural network structure itself
2. attention <-- e.g.: look at entire sentence and learn from data which words to pay attention to when translating into a target language word
3. self-attention = attention but also context of other words in the input sentence to help disambiguate the polysemic meaning of a word

- embedding = words with similar usages/meanings get similar coordinates ([but by itself can't differentiate Apple fruit vs Apple brand](https://www.youtube.com/watch?v=qaWMOYf4ri8))
- positional encoding = word order (https://www.youtube.com/watch?v=qaWMOYf4ri8)
- attention = context ("gravity pull" from other words in sentence https://www.youtube.com/watch?v=qaWMOYf4ri8)
- multi-head attention = multiple weighted contributions of context
- feed-forward / linear = digest
- softmax = probabilities of possibilities of next word in the target output language for example

## History

[ANN ("1") -> DNN ("layers") -> RNN ("short-term memory") -> LSTM ("long-term memory") -> GRU ("simplified LSTM")](https://www.cloudskillsboost.google/course_sessions/6505024/video/363229) -> Encoder-Decoder (to handle sequence-to-sequence tasks) + Attention -> Transformer ("add self-attention") + Positional encoding.

[Transformers process "all" the words _at the same time_, kinda like the heptapod aliens in the movie Arrival](https://www.youtube.com/live/FduFIwExZ0w?si=2wi3QHS80KAtV_ps&t=1677).

## Attention in transformers explanation by 3Blue1Brown

https://www.youtube.com/watch?v=eMlx5fFNoYc
