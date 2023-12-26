from __future__ import annotations

from typing import Tuple

from layers.layers import Layer
from tensor import Tensor


class ReLU(Layer):
    def __call__(self, x: Tensor) -> Tensor:
        result = x * (x > 0)
        result.inputs = (x,)

        def grad_fn(output_grad: Tensor) -> tuple[Tensor | None, ...]:
            x_grad = (result > 0) * output_grad if x.requires_grad else None
            return (x_grad,)

        result.grad_fn = grad_fn
        return result
