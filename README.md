Creash course in LLM.

Week 1 — Perceptron and linear models

Concepts

Binary classification, decision boundaries

Perceptron learning rule

Linear separability, why XOR breaks it

Python deliverables

Numpy perceptron for 2D toy data

Visualize decision boundary + training accuracy

Implement XOR dataset and show failure case

Mini-project

Classify points in 2D (linearly separable) + plot boundary.

Week 2 — Gradients and optimization (the engine of deep learning)

Concepts

Loss functions (MSE, cross-entropy)

Gradients, chain rule, computational graphs (informal)

Gradient descent vs SGD, learning rate, momentum (optional)

Python deliverables

Implement linear regression with MSE and derive gradients by hand

Add logistic regression with sigmoid + cross-entropy

Gradient checking with finite differences (very important habit)

Mini-project

Train logistic regression on a simple dataset (e.g., 2D moons or synthetic blobs).

Week 3 — “Neural network from scratch”: MLP (deep feedforward)

Concepts

Why depth helps (features, compositions)

Activations (ReLU, tanh), vanishing gradients intuition

Backpropagation in an MLP

Python deliverables

Build an MLP in Numpy:

Linear layer, activation, forward pass

Backward pass for each component

Train on MNIST-like small digits (or sklearn digits) and track loss curves

Mini-project

A 2–3 layer MLP classifier with decent accuracy + learning curve plot.

Week 4 — PyTorch version (same MLP, modern workflow)

Concepts

Autograd, tensors, modules, optimizers

Batching, dataloaders, train/eval mode

Overfitting, regularization (weight decay, dropout)

Python deliverables

Rewrite the Week 3 MLP in PyTorch

Add checkpoint saving/loading

Add a simple experiment config pattern (args or dataclass)

Mini-project

Reproduce results quickly and compare Numpy vs PyTorch speed + correctness.

Week 5 — Sequence modeling: RNNs

Concepts

Why feedforward nets struggle with sequences

Recurrent computation, hidden state

Teacher forcing, unrolling through time

Vanishing/exploding gradients in RNNs

Python deliverables

Character-level dataset prep (tiny Shakespeare or your own text)

Implement a vanilla RNN cell from scratch (Numpy or PyTorch)

Train to predict next character, sample text

Mini-project

Generate text from a trained character-level RNN.

Week 6 — LSTMs (fixing long-term dependency issues)

Concepts

Gates: input/forget/output, cell state vs hidden state

Why LSTMs help gradient flow

Practical training tips (clipping, initialization)

Python deliverables

Implement an LSTM cell (from scratch or PyTorch first, then from scratch)

Compare RNN vs LSTM on the same char dataset:

Loss curves

Sample quality

Add gradient clipping and show effect

Mini-project

LSTM text generator that noticeably beats vanilla RNN.

Week 7 — Word embeddings + attention (bridge to Transformers/LLMs)

Concepts

Tokenization basics (char vs word vs subword)

Embeddings, positional information

Attention idea: weighted mixing of context

Python deliverables

Implement word2vec-style skip-gram (optional) or just embedding lookup + training

Implement scaled dot-product attention as a standalone function

Show attention weights on a toy task

Mini-project

A toy “copy” or “reverse sequence” task with attention beating pure recurrence.

Week 8 — Transformer fundamentals (the core of LLMs)

Concepts

Self-attention, multi-head attention

Positional encodings (sinusoidal or learned)

LayerNorm, residual connections, MLP blocks

Causal masking for language modeling

Python deliverables

Implement a single Transformer block in PyTorch

Build a tiny GPT-like model:

Token embedding + positional embedding

N Transformer blocks

LM head

Train on a small dataset (tiny text) and generate samples

Mini-project

“BabyGPT”: trainable end-to-end causal language model that can sample coherent-ish text.

Week 9 — “LLM practice”: training, evaluation, and inference basics

Concepts

Loss = next-token cross entropy

Perplexity, train/val split, early stopping

Sampling strategies: greedy, temperature, top-k, top-p

Context window, KV cache (conceptually)

Python deliverables

Add validation perplexity tracking

Implement sampling knobs (temperature, top-k, top-p)

Save/load model + run inference script

Mini-project

Generate text with controlled sampling and compare behaviors.

Week 10+ — Scaling and “real LLM” ecosystem skills

Concepts

Subword tokenizers (BPE), dataset pipelines

Mixed precision, gradient accumulation

Fine-tuning vs prompt-only, LoRA (optional)

Safety + eval basics (toxicity, leakage, memorization)

Python deliverables

Integrate a BPE tokenizer (e.g., sentencepiece or tiktoken-style equivalents)

Train a slightly larger transformer (still small, but meaningful)

Optional: fine-tune on a narrow domain dataset

Mini-project

A small instruction-tuned or domain-adapted model (even if tiny).
