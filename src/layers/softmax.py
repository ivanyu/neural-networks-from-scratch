from __future__ import annotations

import numpy as np

from layers.layers import Layer
from tensor import Tensor


class SoftMax(Layer):
    def __call__(self, x: Tensor) -> Tensor:
        if x.size == 0:
            raise ValueError("Input size must be at least 1")
        if len(x.shape) != 2:
            raise ValueError("Input must be two-dimensional")

        def f(arr: np.ndarray) -> np.ndarray:
            # Care about numerical stability here.
            e = np.exp(arr - np.max(arr))
            return e / e.sum()

        r = np.apply_along_axis(f, axis=1, arr=x.data)

        result = Tensor(r, requires_grad=x.requires_grad)
        result.inputs = (x,)

        def grad_fn(output_grad: Tensor) -> tuple[Tensor | None, ...]:
            x_grad: Tensor | None = None
            if x.requires_grad:
                x_grad_data = np.zeros_like(r)
                for batch_i in range(r.shape[0]):
                    row = np.expand_dims(r[batch_i], axis=0)
                    a = np.diagflat(row)
                    b = np.matmul(row.T, row)
                    x_grad_data[batch_i] = ((a - b) * output_grad.data[batch_i]).sum(axis=1)
                x_grad = Tensor(x_grad_data)


            return x_grad,

        result.grad_fn = grad_fn
        return result
