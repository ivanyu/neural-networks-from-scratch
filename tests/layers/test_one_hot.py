import pytest

from layers.one_hot import OneHot
from tensor import Tensor


@pytest.mark.parametrize("tensor, expected", [
    (Tensor([0, 1, 2, 2, 1, 0]), Tensor([[
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ]])),
])
def test_forward(tensor: Tensor, expected: Tensor) -> None:
    oh = OneHot(num_classes=3)
    r = oh(tensor)
    assert (r.data == expected.data).all()
