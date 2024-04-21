Adaptation of AlphaZero for the purposes of Tron (CS1410 final project)

To sum up, the model trains through self play using Monte-Carlo Tree Search

Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
https://arxiv.org/abs/1712.01815

Unfortunately we ultimately realized that we didn't have a strong enough GPU or compute to make an implementation that would get a passing grade (Google's implementation, for example, uses millions
worth of compute to make their implementation of AlphaZero).
(We built a simple MinMax tree search with AB-pruning in the last 2 days), but the fact that the model with a week of training on our own computers consistently beat the random and wall-bots is proof
that with enough training time, our model was continuing to learn.
