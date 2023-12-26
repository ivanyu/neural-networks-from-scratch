from typing import Iterable

from tensor import Tensor


class SGD:
    def __init__(self, params: Iterable[Tensor], lr: float) -> None:
        self._params = list(params)
        self._lr = lr

    def step(self) -> None:
        for p in self._params:
            p -= p.grad * self._lr

    def zero_grad(self) -> None:
        for p in self._params:
            p.grad = None
