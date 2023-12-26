# Neural networks from scratch

Building neural networks from the first principle.

## Currently implemented

In this repo the following is implemented:

1. Tensors (Numpy-based) with auto-differentiation.
2. Neural modules:
   - linear with bias;
   - ReLU;
   - softmax;
   - binary cross-entropy loss;
   - mean squared error loss;
   - one-hot encoder;
3. Stochastic gradient descent optimizer.

Enough to implement a simple multilayer perceptron, see [src/example.py](src/example.py).

## Planned

1. Multi-device support for tensors and CUDA support.
2. More neural modules:
   - convolutions;
   - batch normalization;
   - drop out.
3. Experiment with JIT.
