import numpy as np
import pytest

from layers.softmax import SoftMax
from tensor import Tensor


def test_forward_empty() -> None:
    sm = SoftMax()
    with pytest.raises(ValueError, match="Input size must be at least 1"):
        sm(Tensor([]))


@pytest.mark.parametrize("tensor", [
    Tensor([0.0]), Tensor([[[0.0], [0.0]], [[0.0], [0.0]]]),
])
def test_forward_wrong_dimensions(tensor: Tensor) -> None:
    sm = SoftMax()
    with pytest.raises(ValueError, match="Input must be two-dimensional"):
        sm(tensor)


@pytest.mark.parametrize("tensor, expected", [
    (Tensor([[10.0]]), Tensor([[1.0]])),
    (Tensor([
        [0.0, 1.0],
        [2.0, 5.0]
    ]), Tensor([
        [0.2689, 0.7311],
        [0.0474, 0.9526]
    ])),
])
@pytest.mark.parametrize("requires_grad", [True, False])
def test_forward(tensor: Tensor, expected: Tensor, requires_grad: bool) -> None:
    sm = SoftMax()
    tensor.requires_grad = requires_grad
    y = sm(tensor)
    assert np.isclose(y.data, expected.data, rtol=0.001).all()
    assert np.isclose(y.data.sum(axis=1), np.array([1.0] * tensor.shape[0])).all()
    assert y.requires_grad == tensor.requires_grad


def test_backward() -> None:
    x_data = np.array([
        [2.0, 0.1, 3.0],
        [0.1, 0.1, 100.0],
    ])
    x = Tensor(x_data, requires_grad=True, retain_grad=True)

    softmax = SoftMax()
    probs = softmax(x)

    prob_upstream_grad = np.array([
        [-2.0, 1.0, -3.0],
        [-2.0, -2.0, 0.1],
    ])
    probs.backward(Tensor(prob_upstream_grad))

    # See https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    # https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    x_grad_expected = np.zeros((2, 3))
    for batch_i in range(2):
        probs_vec = probs.data[batch_i].reshape(3, )
        tmp = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if i == j:
                    tmp[i, j] = probs_vec[i] * (1 - probs_vec[i])
                else:
                    tmp[i, j] = -probs_vec[i] * probs_vec[j]
        x_grad_expected[batch_i] = (tmp * prob_upstream_grad[batch_i]).sum(axis=1)

    assert np.isclose(x.grad.data, x_grad_expected).all()
