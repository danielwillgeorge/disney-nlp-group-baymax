# Baymax: A Recurrent Neural Network for Question Answering

![Baymax: A Recurrent Neural Network for Question Answering](https://i.imgur.com/g5PvcFT.jpg)

Baymax is based on a Bidirectional Attention Flow (BiDAF) model.

## Model Description

### Embedding Layer (layers.Embedding)

We produce embeddings for the context paragraph (for example, an abstract) and the question being asked.  We use a two-step process:
   1. We project each embedding to have dimensionality equal to the hidden size of the model.
   2. We apply a Highway Network to refine the embedded representation.

### Encoder Layer (layers.RNNEncoder)

The encoder layer uses the embedding layer's output as its input, and uses a bidirectional LSTM to allow the model to incorporate dependencies between timesteps of the embedding layer's output.

### Attention Layer (layers.BiDAFAttention)

This is the core of the BiDAF model.  The main idea of this layer is that attention flows from the context (abstract) to the question, and from the question to the context.
We compute a similarity matrix, which contains a similarity score for each pair of context and question hidden states.  Then, we perform Context-to-Question (C2Q) Attention
followed by Question-to-Context (Q2C) Attention.

### Modeling Layer (layers.RNNEncoder)

This layer refines the vectors after the attention layer.

### Output Layer (layers.BiDAFOutput)

This layer is tasked with assigning probabilities to each position in the context.  Specifically, we compute the probability that the answer begins and ends at some positions i and j.

## Training Details

### Loss Function

We use the sum of the negative log-likelihood (cross-entropy) loss for the start and end locations.  We use the Adadelta optimizer to minimize loss.

### GPU

Training was completed on the Google Cloud Platform (GCP) Compute Engine using a NVIDIA Tesla P4 GPU.

## Acknowledgements/Credits

This approach follows the original BiDAF paper, with code adapted from the work of Chris Chute (Stanford University) and Mina Lee (Stanford University).