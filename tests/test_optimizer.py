import numpy as np

from optimizer import SGD
from tensor import Tensor


def test_step() -> None:
    t1 = Tensor(np.array([10.0, 20.0, 30.0]))
    t1.grad = Tensor(np.array([1.0, -2.0, 3.0]))

    t2 = Tensor(np.array([
        [100.0, 200.0],
        [300.0, 400.0],
    ]))
    t2.grad = Tensor(np.array([[0.0, 0.0], [0.0, 0.1]]))

    optimizer = SGD([t1, t2], lr=0.1)
    optimizer.step()

    expected_t1_after_1_step = Tensor(np.array(
        [10.0 - 0.1 * 1.0, 20.0 - 0.1 * (-2.0), 30.0 - 0.1 * 3.0]
    ))
    expected_t2_after_1_step = Tensor(np.array([
        [100.0, 200.0],
        [300.0, 400.0 - 0.1 * 0.1]
    ]))

    assert (t1.data == expected_t1_after_1_step.data).all()
    assert (t2.data == expected_t2_after_1_step.data).all()


def test_zero_grad() -> None:
    t1 = Tensor(np.array([10.0, 20.0, 30.0]))
    t1.grad = Tensor(np.array([1.0, -2.0, 3.0]))

    t2 = Tensor(np.array([
        [100.0, 200.0],
        [300.0, 400.0],
    ]))
    t2.grad = Tensor(np.array([[0.0, 0.0], [0.0, 0.1]]))

    optimizer = SGD([t1, t2], lr=0.1)
    optimizer.zero_grad()

    assert t1.grad is None
    assert t2.grad is None
