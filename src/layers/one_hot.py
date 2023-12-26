from __future__ import annotations

import numpy as np

from layers.layers import Layer
from tensor import Tensor


class OneHot(Layer):
    def __init__(self, num_classes: int) -> None:
        self._num_classes = num_classes

    def __call__(self, x: Tensor) -> Tensor:
        if len(x.shape) != 1:
            raise ValueError(f"Incorrect shape {x.shape}")
        r = Tensor.zeros((x.shape[0], self._num_classes))
        # TODO get rid of direct np usage
        r.data[np.arange(x.shape[0]), x.data] = 1
        return Tensor(r, requires_grad=x.requires_grad)
