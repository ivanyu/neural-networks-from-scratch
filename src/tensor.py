from __future__ import annotations

from typing import Any, Protocol, Collection

import numpy as np
from typing_extensions import override


Shape = tuple[int, ...]


class GradFn(Protocol):
    def __call__(self, output_grad: Tensor) -> tuple[Tensor, ...]:
        ...


class Tensor:
    def __init__(self, data: np.ndarray | Collection | Tensor, requires_grad: bool = False, retain_grad: bool = False) -> None:
        match data:
            case Tensor():
                self.data = data.data
            case _:
                self.data = np.array(data)

        self.grad: Tensor | None = None
        self.grad_fn: GradFn | None = None
        self.requires_grad = requires_grad
        self.retain_grad = retain_grad

        self.inputs: tuple[Tensor, ...] = tuple()

    @override
    def __repr__(self) -> str:
        return f"Tensor({repr(self.data)})"

    @property
    def shape(self) -> Shape:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    def __add__(self, other: Any) -> Tensor:
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data, requires_grad=self.requires_grad, retain_grad=False)
        elif isinstance(other, np.ndarray):
            return Tensor(self.data + other, requires_grad=self.requires_grad, retain_grad=False)
        else:
            raise ValueError("other must be Tensor or numpy.array")

    def __sub__(self, other: Any) -> Tensor:
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data, requires_grad=self.requires_grad, retain_grad=False)
        elif isinstance(other, np.ndarray):
            return Tensor(self.data - other, requires_grad=self.requires_grad, retain_grad=False)
        else:
            raise ValueError("other must be Tensor or numpy.array")

    def __isub__(self, other: Any) -> Tensor:
        # TODO test
        if isinstance(other, Tensor):
            self.data -= other.data
            return self
        elif isinstance(other, np.ndarray):
            self.data -= other
            return self
        else:
            raise ValueError("other must be Tensor or numpy.array")

    def __mul__(self, other: Any) -> Tensor:
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data, requires_grad=self.requires_grad, retain_grad=False)
        elif isinstance(other, np.ndarray) or isinstance(other, float) or isinstance(other, int):
            return Tensor(self.data * other, requires_grad=self.requires_grad, retain_grad=False)
        else:
            raise ValueError("other must be Tensor, numpy.array, float, or int")

    def __truediv__(self, other: Any) -> Tensor:
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data, requires_grad=self.requires_grad, retain_grad=False)
        elif isinstance(other, np.ndarray) or isinstance(other, float) or isinstance(other, int):
            return Tensor(self.data / other, requires_grad=self.requires_grad, retain_grad=False)
        else:
            raise ValueError("other must be Tensor, numpy.array, float, or int")

    def __neg__(self) -> Tensor:
        return Tensor(-self.data, requires_grad=self.requires_grad, retain_grad=False)

    def __gt__(self, other: Any) -> Tensor:
        if isinstance(other, float) or isinstance(other, int):
            return Tensor(self.data > other, requires_grad=self.requires_grad, retain_grad=False)
        else:
            raise ValueError("other must be float or int")

    def sum(self, axis: int) -> Tensor:
        return Tensor([self.data.sum(axis=axis)], requires_grad=self.requires_grad, retain_grad=False)

    def mean(self) -> Tensor:
        return Tensor([np.mean(self.data)], requires_grad=self.requires_grad, retain_grad=False)

    @property
    def T(self) -> Tensor:
        return Tensor(self.data.T, requires_grad=self.requires_grad, retain_grad=False)

    def reshape(self, shape: Shape) -> Tensor:
        return Tensor(self.data.reshape(shape), requires_grad=self.requires_grad, retain_grad=False)

    @staticmethod
    def matmul(a: Tensor, b: Tensor) -> Tensor:
        if not isinstance(a, Tensor):
            raise ValueError("a must be Tensor")
        if not isinstance(b, Tensor):
            raise ValueError("b must be Tensor")
        return Tensor(np.matmul(a.data, b.data), requires_grad=a.requires_grad or b.requires_grad, retain_grad=False)

    @staticmethod
    def zeros(shape: Shape) -> Tensor:
        return Tensor(np.zeros(shape), retain_grad=False)

    @staticmethod
    def zeros_like(other: Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            raise ValueError("other must be Tensor")
        return Tensor(np.zeros_like(other.data), retain_grad=False)

    @property
    def is_leaf(self) -> bool:
        return not self.inputs

    @staticmethod
    def rand(*args: int) -> Tensor:
        return Tensor(np.random.rand(*args), retain_grad=False)

    def backward(self, upstream_grad: Tensor | None = None) -> None:
        # TODO assert dimensions

        if upstream_grad is None:
            upstream_grad = Tensor([1.0])

        topology = self._build_topology()
        assert topology[0].tensor is self
        topology[0].grad = upstream_grad
        for t in topology:
            if t.grad_fn is not None:
                grads = t.grad_fn(t.grad)
                for inp, grad in zip(t.inputs, grads):
                    if grad is not None:
                        inp.grad += grad

        # Retain gradients
        for t in topology:
            t.retain_grad_if_needed()

    def _build_topology(self) -> list[_TensorNode]:
        visited: set[_TensorNode] = set()
        stack: list[_TensorNode] = []

        def topo(t: _TensorNode) -> None:
            visited.add(t)
            for i in t.inputs:
                i.outputs.append(t)
                if i.grad_fn is None and not i.is_leaf:
                    raise ValueError("grad_fn is None for non-leaf tensor")
                if i not in visited:
                    topo(i)
            stack.insert(0, t)

        topo(_TensorNode(self))

        return stack


class _TensorNode:
    def __init__(self, tensor: Tensor) -> None:
        self.tensor = tensor
        self.inputs = [_TensorNode(i) for i in tensor.inputs]
        self.outputs: list[_TensorNode] = []
        self.grad = Tensor.zeros_like(tensor)

    @property
    def grad_fn(self) -> GradFn:
        return self.tensor.grad_fn

    @property
    def retain_grad(self) -> bool:
        return self.tensor.retain_grad

    @property
    def is_leaf(self) -> bool:
        return self.tensor.is_leaf

    def retain_grad_if_needed(self) -> None:
        if self.retain_grad and self.grad is not None:
            self.tensor.grad = self.grad
