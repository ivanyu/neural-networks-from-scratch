import numpy as np

from tensor import Tensor
from layers.linear import Linear
from layers.relu import ReLU
from layers.loss import MSELoss


def test_model_1() -> None:
    x_data = np.array([
        [5.6000, 2.9000, 3.6000, 1.3000],
        [4.8000, 3.1000, 1.6000, 0.2000],
        [5.6000, 2.8000, 4.9000, 2.0000],
    ])
    x = Tensor(x_data)
    x.requires_grad = True
    x.retain_grad = True

    target_data = np.array([
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 0., 1.],
    ])
    target = Tensor(target_data)
    target.requires_grad = True
    target.retain_grad = True

    linear_1_weight_data = np.array([
        [-0.0419, -0.0171, -0.1875, 0.1150],
        [-0.2861, -0.0882, 0.1938, 0.4693],
        [0.1178, -0.1696, 0.0479, -0.0560],
        [0.2041, 0.0573, 0.1959, 0.4849],
        [-0.2076, -0.0177, 0.1150, -0.0033],
        [-0.0479, -0.4425, -0.4313, -0.4499],
        [-0.4892, -0.4657, -0.3788, -0.4510],
        [-0.4690, 0.2192, 0.3067, 0.3379],
    ])
    linear_1_bias_data = np.array([0.2694, 0.1694, 0.2203, -0.2765, 0.4502, -0.0345, 0.4314, 0.1533])
    linear_1 = Linear(4, 8)
    linear_1.set_weight(Tensor(linear_1_weight_data))
    linear_1.set_bias(Tensor(linear_1_bias_data))
    linear_1.weight.retain_grad = True
    linear_1.bias.retain_grad = True

    linear_2_weight_data = np.array([
        [0.2767, 0.2820, -0.0739, -0.1028, 0.0532, -0.0150, 0.0553, 0.1794],
        [-0.2763, -0.0162, -0.2775, 0.3414, -0.2487, 0.0676, -0.0966, 0.2010],
        [0.0012, -0.0356, 0.2588, 0.3229, -0.2566, -0.3410, 0.0295, 0.1114],
    ])
    linear_2_bias_data = np.array([0.0807, 0.3266, 0.1587])
    linear_2 = Linear(8, 3)
    linear_2.set_weight(Tensor(linear_2_weight_data))
    linear_2.set_bias(Tensor(linear_2_bias_data))
    linear_2.weight.retain_grad = True
    linear_2.bias.retain_grad = True

    relu_2 = ReLU()
    loss = MSELoss()

    x1_expected = np.zeros((3, 8))
    for batch_i in range(x_data.shape[0]):  # iterate over batch
        for weight_i in range(linear_1_weight_data.shape[0]):
            for j in range(x_data.shape[1]):
                x1_expected[batch_i][weight_i] += x_data[batch_i][j] * linear_1_weight_data[weight_i][j]
            x1_expected[batch_i][weight_i] += linear_1_bias_data[weight_i]
    x1 = linear_1(x)
    x1.retain_grad = True
    assert np.isclose(x1_expected, x1.data).all()

    x2_expected = np.zeros((3, 3))
    for batch_i in range(x1.data.shape[0]):  # iterate over batch
        for weight_i in range(linear_2_weight_data.shape[0]):
            for j in range(x1.data.shape[1]):
                x2_expected[batch_i][weight_i] += x1.data[batch_i][j] * linear_2_weight_data[weight_i][j]
            x2_expected[batch_i][weight_i] += linear_2_bias_data[weight_i]
    x2 = linear_2(x1)
    x2.retain_grad = True
    assert np.isclose(x2_expected, x2.data).all()

    x3_expected = np.array(x2.data)
    for i in range(x3_expected.shape[0]):
        for j in range(x3_expected.shape[1]):
            if x3_expected[i][j] < 0:
                x3_expected[i][j] = 0.0
    x3 = relu_2(x2)
    x3.retain_grad = True
    assert np.isclose(x3_expected, x3.data).all()

    L_expected = ((x3.data[0][0] - target_data[0][0]) ** 2
                  + (x3.data[0][1] - target_data[0][1]) ** 2
                  + (x3.data[0][2] - target_data[0][2]) ** 2
                  + (x3.data[1][0] - target_data[1][0]) ** 2
                  + (x3.data[1][1] - target_data[1][1]) ** 2
                  + (x3.data[1][2] - target_data[1][2]) ** 2
                  + (x3.data[2][0] - target_data[2][0]) ** 2
                  + (x3.data[2][1] - target_data[2][1]) ** 2
                  + (x3.data[2][2] - target_data[2][2]) ** 2
                  ) / 9
    L = loss(x3, target)
    L.retain_grad = True
    assert L.data[0] == L_expected

    L.backward()

    L_grad_expected = np.array([1.0])
    assert (L.grad.data == L_grad_expected).all()

    x3_grad_expected = np.zeros((3, 3))
    target_grad_expected = np.zeros((3, 3))
    for i in range(x3.shape[0]):
        for j in range(x3.shape[1]):
            x3_grad_expected[i][j] = (x3.data[i][j] - target_data[i][j]) / (9 / 2)
            x3_grad_expected[i][j] *= L_grad_expected[0]

            target_grad_expected[i][j] = (-1) * x3_grad_expected[i][j]
            target_grad_expected[i][j] *= L_grad_expected[0]

    assert (x3.grad.data == x3_grad_expected).all()
    assert (target.grad.data == target_grad_expected).all()

    x2_grad_expected = np.zeros((3, 3))
    for i in range(x2.shape[0]):
        for j in range(x2.shape[1]):
            if x2.data[i][j] > 0:
                x2_grad_expected[i][j] = 1
            else:
                x2_grad_expected[i][j] = 0
            x2_grad_expected[i][j] *= x3_grad_expected[i][j]

    assert (x2.grad.data == x2_grad_expected).all()

    # See e.g. https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html
    linear_2_weight_grad_expected = np.zeros((3, 8))
    linear_2_bias_grad_expected = np.zeros((1, 3))
    x1_grad_expected = np.zeros((3, 8))
    for batch_i in range(x2.shape[0]):
        weight_grad_per_batch_element = np.zeros((3, 8))
        bias_grad_per_batch_element = np.zeros((1, 3))
        x_grad_per_batch_element = np.zeros((8,))

        for row in range(x2.shape[1]):
            for col in range(x1.shape[1]):
                # Calculate per batch element and aggregate in the end (by summing or stacking).
                weight_grad_per_batch_element[row][col] = x2.grad.data[batch_i][row] * x1.data[batch_i][col]
                x_grad_per_batch_element[col] += x2.grad.data[batch_i][row] * linear_2_weight_data[row][col]

            bias_grad_per_batch_element[0][row] = 1.0 * x2.grad.data[batch_i][row]

        linear_2_weight_grad_expected += weight_grad_per_batch_element
        linear_2_bias_grad_expected += bias_grad_per_batch_element
        x1_grad_expected[batch_i] = x_grad_per_batch_element
    assert np.isclose(linear_2.weight.grad.data, linear_2_weight_grad_expected).all()
    assert np.isclose(linear_2.bias.grad.data, linear_2_bias_grad_expected).all()
    assert np.isclose(x1.grad.data, x1_grad_expected).all()

    linear_1_weight_grad_expected = np.zeros((8, 4))
    linear_1_bias_grad_expected = np.zeros((1, 8))
    x_grad_expected = np.zeros((3, 4))
    for batch_i in range(x1.shape[0]):
        weight_grad_per_batch_element = np.zeros((8, 4))
        bias_grad_per_batch_element = np.zeros((1, 8))
        x_grad_per_batch_element = np.zeros((4,))

        for row in range(x1.shape[1]):
            for col in range(x.shape[1]):
                # Calculate per batch element and aggregate in the end (by summing or stacking).
                weight_grad_per_batch_element[row][col] = x1.grad.data[batch_i][row] * x.data[batch_i][col]
                x_grad_per_batch_element[col] += x1.grad.data[batch_i][row] * linear_1_weight_data[row][col]

            bias_grad_per_batch_element[0][row] = 1.0 * x1.grad.data[batch_i][row]

        linear_1_weight_grad_expected += weight_grad_per_batch_element
        linear_1_bias_grad_expected += bias_grad_per_batch_element
        x_grad_expected[batch_i] = x_grad_per_batch_element
    assert np.isclose(linear_1.weight.grad.data, linear_1_weight_grad_expected).all()
    assert np.isclose(linear_1.bias.grad.data, linear_1_bias_grad_expected).all()
    assert np.isclose(x.grad.data, x_grad_expected).all()
