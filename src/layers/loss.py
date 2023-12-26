from __future__ import annotations

import numpy as np

from layers.layers import Layer
from tensor import Tensor


class BinaryCrossEntropyLoss(Layer):
    def __call__(self, x: Tensor, target: Tensor) -> Tensor:
        if x.shape != target.shape:
            raise ValueError(f"Input and target shapes must be equal, but {x.shape} and {target.shape} given")

        t = target.data
        one_minus_t = 1 - t
        p = x.data
        one_minus_p = 1 - p

        log_p = np.clip(a=np.log(p), a_min=-100.0, a_max=1.0)
        log_one_minus_p = np.clip(a=np.log(one_minus_p), a_min=-100.0, a_max=1.0)
        r = -(t * log_p + one_minus_t * log_one_minus_p)
        result = Tensor([np.mean(r)], requires_grad=x.requires_grad or target.requires_grad)
        result.inputs = (x, target)

        def grad_fn(output_grad: Tensor) -> tuple[Tensor | None, ...]:
            x_grad: Tensor | None = None
            if x.requires_grad:
                x_grad = Tensor((t / p - one_minus_t / one_minus_p) / (-x.size) * output_grad.data)

            target_grad: Tensor | None = None
            if target.requires_grad:
                target_grad = Tensor((log_p - log_one_minus_p) / (-x.size) * output_grad.data)

            return x_grad, target_grad

        result.grad_fn = grad_fn
        return result


class MSELoss(Layer):
    def __call__(self, x: Tensor, target: Tensor) -> Tensor:
        if x.shape != target.shape:
            raise ValueError(f"Input and target shapes must be equal, but {x.shape} and {target.shape} given")
        t = x - target
        t_square = t * t
        result = t_square.mean()
        result.requires_grad = x.requires_grad or target.requires_grad
        result.inputs = (x, target)

        def grad_fn(output_grad: Tensor) -> tuple[Tensor | None, ...]:
            # After the derivation, the square decreases the denominator 2x.
            x_grad = (t / (t.size / 2) * output_grad) if x.requires_grad else None
            target_grad = -x_grad if target.requires_grad else None
            return x_grad, target_grad

        result.grad_fn = grad_fn
        return result
