from __future__ import annotations

from typing import Tuple

from layers.layers import Layer
from tensor import Tensor


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weight: Tensor = Tensor.rand(out_features, in_features)
        self.weight.requires_grad = True
        self.weight.retain_grad = True
        self.bias: Tensor = Tensor.rand(out_features)
        self.bias.requires_grad = True
        self.bias.retain_grad = True

    def set_weight(self, weight: Tensor) -> None:
        expected_shape = (self.out_features, self.in_features)
        if weight.shape != expected_shape:
            raise ValueError(f"Invalid shape: {weight.shape}, expected: {expected_shape}")
        self.weight = Tensor(weight, requires_grad=True, retain_grad=True)

    def set_bias(self, bias: Tensor) -> None:
        expected_shape = (self.out_features,)
        if bias.shape != expected_shape:
            raise ValueError(f"Invalid shape: {bias.shape}, expected: {expected_shape}")
        self.bias = Tensor(bias, requires_grad=True, retain_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        expected_size = (self.in_features,)
        shape_without_batch_dim = x.shape[1:]
        if shape_without_batch_dim != expected_size:
            raise ValueError(
                f"Invalid shape (without batch dimension): {shape_without_batch_dim}, expected: {expected_size}"
            )

        result = Tensor.matmul(x, self.weight.T) + self.bias
        result.inputs = (x, self.weight, self.bias)

        def grad_fn(output_grad: Tensor) -> tuple[Tensor, ...]:
            x_grad = Tensor.matmul(output_grad, self.weight) if x.requires_grad else None
            weight_grad = Tensor.matmul(output_grad.T, x) if self.weight.requires_grad else None
            bias_grad = output_grad.sum(axis=0).reshape(self.bias.shape) if self.bias.requires_grad else None
            return x_grad, weight_grad, bias_grad

        result.grad_fn = grad_fn
        return result
