# MNIST Digit Classification — Neural Network Hyperparameter Study

A systematic comparison of feedforward neural network configurations for handwritten digit classification using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The goal was to isolate the effect of key hyperparameters on model performance through controlled experiments.

## Overview

Two activation functions (ReLU and Tanh) were evaluated across a range of hyperparameter configurations:

- **Optimizers:** SGD (with momentum) vs. Adam
- **Learning rates:** 0.01 vs. 0.001
- **Batch sizes:** 32 vs. 64
- **Regularization:** L2 penalty, early stopping, and their combination

Each variable was tested independently to observe its isolated effect on training stability, validation accuracy, and overfitting behaviour.

## Model Architecture

All models use a sequential feedforward architecture:

| Layer | Units | Activation |
|-------|-------|------------|
| Input | 784 | — |
| Hidden 1 | 128 | ReLU or Tanh |
| Hidden 2 | 64 | ReLU or Tanh |
| Hidden 3 | 32 | ReLU or Tanh |
| Output | 10 | Softmax |

Input images (28×28 grayscale) are normalized to [0, 1] and flattened to 784-dimensional vectors. Labels are one-hot encoded.

## Key Findings

- **SGD outperformed Adam** on training stability for both activation functions at the same learning rate
- **Adam required a lower learning rate** (0.001 vs. 0.01) to converge reliably, particularly with Tanh
- **Batch size 64** provided the best trade-off between gradient stability and training speed
- **L2 regularization** (rate=0.001) reduced overfitting with minimal accuracy cost; combining it with early stopping offered no additional benefit for this architecture
- **Best model:** ReLU + SGD (lr=0.01, batch=64) — **97.18% test accuracy**

## Project Structure

```
mnist-nn-comparison/
├── README.md
├── requirements.txt
└── mnist_neural_network_comparison.ipynb
```

## Requirements

```
tensorflow>=2.10
numpy>=1.23
matplotlib>=3.6
```

Install with:

```bash
pip install -r requirements.txt
```

## Running the Notebook

```bash
git clone https://github.com/mkearney20/mnist-nn-comparison.git
cd mnist-nn-comparison
pip install -r requirements.txt
jupyter notebook mnist_neural_network_comparison.ipynb
```

Run all cells from top to bottom. The MNIST dataset is downloaded automatically via `tensorflow.keras.datasets`.
