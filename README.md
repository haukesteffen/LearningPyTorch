# LearningPyTorch

This repository is inspired by and loosely based on Karpathy's [makemore](https://github.com/karpathy/makemore). I use it to get a better understanding of PyTorch and grasp various concepts in NLP. The idea is to train language models of varying complexity using a text corpus of [Hacker News Comments](https://github.com/haukesteffen/HNPulse/blob/main/src/scrape.py).

1. [Simple bigram model](https://github.com/haukesteffen/LearningPyTorch/blob/main/1-bigram-model.ipynb). Counts bigram occurrences and uses multinomial sampling to predict the next character in the sequence.
2. [Simple neural network](https://github.com/haukesteffen/LearningPyTorch/blob/main/2-simple-nn.ipynb). Using a simple neural network to approximate the bigram distribution explicitly learned in the previous model.
3. [Neural network with input embeddings](https://github.com/haukesteffen/LearningPyTorch/blob/main/3-embeddings-nn.ipynb). Introducing input embeddings to improve model performance.
4. [Recurrent neural network](https://github.com/haukesteffen/LearningPyTorch/blob/main/4-rnn.ipynb). Using a recurrent neural network to further lower negative log likelihood.
5. LSTM/GLU. to-do
6. Transformer. to-do

Currently can't get the low-level PyTorch implementation of RNNs to work. The problem seems to be a lack of parallelization, so training takes forever. However, I am currently having problems wrapping my head around the parallelization, especially since the batch dimension is of different size for every input sequence. Any advice is appreciated!