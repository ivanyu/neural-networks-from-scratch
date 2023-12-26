import math

import numpy as np

from layers.loss import BinaryCrossEntropyLoss
from tensor import Tensor


def test_binary_cross_entropy_loss_forward_and_backward() -> None:
    probs_data = np.array([
        [0.1, 0.2, 0.7],
        [0.15, 0.05, 0.8],
    ])
    probs = Tensor(probs_data)
    probs.requires_grad = True
    probs.retain_grad = True

    target_data = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    target = Tensor(target_data)
    target.requires_grad = True
    target.retain_grad = True

    bce = BinaryCrossEntropyLoss()
    loss = bce(probs, target)

    loss_expected = 0.0
    for batch_i in range(probs.shape[0]):
        for el_i in range(probs.shape[1]):
            t = target_data[batch_i, el_i]
            p = probs_data[batch_i, el_i]
            log_p = np.clip(a=np.log(p), a_min=-100.0, a_max=1.0)
            log_1_minus_p = np.clip(a=np.log(1 - p), a_min=-100.0, a_max=1.0)
            loss_expected -= (t * log_p + (1 - t) * log_1_minus_p)
    loss_expected /= probs.size

    assert loss.data[0] == loss_expected

    loss.backward()

    probs_grad_expected = np.zeros(probs.shape)
    for batch_i in range(probs.shape[0]):
        for el_i in range(probs.shape[1]):
            t = target_data[batch_i, el_i]
            p = probs_data[batch_i, el_i]
            probs_grad_expected[batch_i, el_i] = t / p - (1 - t) / (1 - p)
            probs_grad_expected[batch_i, el_i] /= -probs.size

    assert np.isclose(probs.grad.data, probs_grad_expected).all()

    target_grad_expected = np.zeros(target.shape)
    for batch_i in range(target.shape[0]):
        for el_i in range(target.shape[1]):
            p = probs_data[batch_i, el_i]
            target_grad_expected[batch_i, el_i] = math.log(p) - math.log(1 - p)
            target_grad_expected[batch_i, el_i] /= -probs.size

    assert np.isclose(target.grad.data, target_grad_expected).all()
