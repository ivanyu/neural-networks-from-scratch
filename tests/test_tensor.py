import pytest

from tensor import Tensor, Shape
import numpy as np


@pytest.mark.parametrize("tensor, expected_shape, expected_size", [
    (Tensor([]), (0, ), 0),
    (Tensor([0, 1, 2, 3]), (4, ), 4),
    (Tensor([[0, 1], [2, 3]]), (2, 2), 4),
])
def test_shape_and_size(tensor: Tensor, expected_shape: Shape, expected_size: int) -> None:
    assert tensor.shape == expected_shape
    assert tensor.size == expected_size


@pytest.mark.parametrize("tensor, target_shape, expected_tensor", [
    (Tensor([]), (0, ), Tensor([])),
    (Tensor([0, 1, 2, 3]), (2, 2), Tensor([[0, 1], [2, 3]])),
    (Tensor([[0, 1], [2, 3]]), (4,), Tensor([0, 1, 2, 3])),
    (Tensor([[0, 1], [2, 3]]), (1, 4), Tensor([[0, 1, 2, 3]])),
])
def test_reshape(tensor: Tensor, target_shape: Shape, expected_tensor: Tensor) -> None:
    reshaped = tensor.reshape(target_shape)
    assert reshaped.shape == expected_tensor.shape
    assert (reshaped.data == expected_tensor.data).all()


@pytest.mark.parametrize("array1, array2", [
    (np.array([0, 1, 2, 3]), np.array([10, 20, 30, 40])),
    (np.array([[0, 1], [2, 3]]), np.array([[10, 20], [30, 40]])),
    (np.array([[0, 1], [2, 3]]), np.array([100])),
])
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("retain_grad", [True, False])
def test_add(array1: np.ndarray, array2: np.ndarray, requires_grad: bool, retain_grad: bool) -> None:
    t1 = Tensor(array1)
    t1.requires_grad = requires_grad
    t1.retain_grad = retain_grad
    t2 = Tensor(array2)

    sum1 = t1 + t2
    sum2 = t1 + t2.data
    assert (np.all(sum1.data == sum2.data))
    assert (np.all(sum1.data == array1 + array2))
    assert sum1.requires_grad is requires_grad
    assert sum2.requires_grad is requires_grad
    assert sum1.retain_grad is False
    assert sum2.retain_grad is False


@pytest.mark.parametrize("array1, array2", [
    (np.array([0, 1, 2, 3]), np.array([10, 20, 30, 40])),
    (np.array([[0, 1], [2, 3]]), np.array([[10, 20], [30, 40]])),
    (np.array([[0, 1], [2, 3]]), np.array([100])),
])
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("retain_grad", [True, False])
def test_sub(array1: np.ndarray, array2: np.ndarray, requires_grad: bool, retain_grad: bool) -> None:
    t1 = Tensor(array1)
    t1.requires_grad = requires_grad
    t1.retain_grad = retain_grad
    t2 = Tensor(array2)

    sub1 = t1 - t2
    sub2 = t1 - t2.data
    assert (np.all(sub1.data == sub2.data))
    assert (np.all(sub1.data == array1 - array2))
    assert sub1.requires_grad is requires_grad
    assert sub1.retain_grad is False
    assert sub2.requires_grad is requires_grad
    assert sub2.retain_grad is False


@pytest.mark.parametrize("array1, array2", [
    (np.array([0, 1, 2, 3]), np.array([10, 20, 30, 40])),
    (np.array([[0, 1], [2, 3]]), np.array([[10, 20], [30, 40]])),
    (np.array([[0, 1], [2, 3]]), np.array([100])),
])
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("retain_grad", [True, False])
def test_isub(array1: np.ndarray, array2: np.ndarray, requires_grad: bool, retain_grad: bool) -> None:
    t1 = Tensor(array1)
    t1.requires_grad = requires_grad
    t1.retain_grad = retain_grad
    t2 = Tensor(array2)
    expected = array1 - array2

    t1 -= t2

    assert (np.all(t1.data == expected))
    assert t1.requires_grad is requires_grad
    assert t1.retain_grad is retain_grad


@pytest.mark.parametrize("a, b", [
    (np.array([0, 1, 2, 3]), np.array([10, 20, 30, 40])),
    (np.array([[0, 1], [2, 3]]), np.array([[10, 20], [30, 40]])),
    (np.array([[0, 1], [2, 3]]), np.array([100])),
    (np.array([[0, 1], [2, 3]]), 100),
    (np.array([[0, 1], [2, 3]]), 100.0),
])
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("retain_grad", [True, False])
def test_mul(a: np.ndarray, b: np.ndarray | float | int, requires_grad: bool, retain_grad: bool) -> None:
    tensor_a = Tensor(a)
    tensor_a.requires_grad = requires_grad
    tensor_a.retain_grad = retain_grad

    results = []
    results.append(a * b)

    mul1 = tensor_a * b
    assert mul1.requires_grad is requires_grad
    assert mul1.retain_grad is False
    results.append(mul1.data)

    if isinstance(b, np.ndarray):
        tensor_b = Tensor(b)
        mul2 = tensor_a * tensor_b
        assert mul2.requires_grad is requires_grad
        assert mul2.retain_grad is False
        results.append(mul2.data)

    for i in range(len(results) - 1):
        assert (results[i] == results[i + 1]).all()


@pytest.mark.parametrize("a, b", [
    (np.array([0, 1, 2, 3]), np.array([10, 20, 30, 40])),
    (np.array([[0, 1], [2, 3]]), np.array([[10, 20], [30, 40]])),
    (np.array([[0, 1], [2, 3]]), np.array([100])),
    (np.array([[0, 1], [2, 3]]), 100),
    (np.array([[0, 1], [2, 3]]), 100.0),
])
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("retain_grad", [True, False])
def test_div(a: np.ndarray, b: np.ndarray | float | int, requires_grad: bool, retain_grad: bool) -> None:
    tensor_a = Tensor(a)
    tensor_a.requires_grad = requires_grad
    tensor_a.retain_grad = retain_grad

    results = []
    results.append(a / b)
    div1 = tensor_a / b
    assert div1.requires_grad is requires_grad
    assert div1.retain_grad is False
    results.append(div1.data)

    if isinstance(b, np.ndarray):
        tensor_b = Tensor(b)
        div2 = tensor_a / tensor_b
        assert div2.requires_grad is requires_grad
        assert div2.retain_grad is False
        results.append(div2.data)

    for i in range(len(results) - 1):
        assert (results[i] == results[i + 1]).all()


@pytest.mark.parametrize("tensor, expected", [
    (Tensor([]), Tensor([])),
    (Tensor([[1.0]]), Tensor([[-1.0]])),
    (Tensor([0, -1, -2, -3]), Tensor([0, 1, 2, 3])),
])
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("retain_grad", [True, False])
def test_neg(tensor: Tensor, expected: Tensor, requires_grad: bool, retain_grad: bool) -> None:
    tensor.requires_grad = requires_grad
    tensor.retain_grad = retain_grad
    neg = -tensor
    assert (neg.data == expected.data).all()
    assert neg.requires_grad is requires_grad
    assert neg.retain_grad is False


@pytest.mark.parametrize("tensor, expected", [
    (Tensor([]), Tensor([])),
    (Tensor([[1.0]]), Tensor([[True]])),
    (Tensor([0, -1, 2, -3]), Tensor([False, False, True, False])),
])
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("retain_grad", [True, False])
def test_gt(tensor: Tensor, expected: Tensor, requires_grad: bool, retain_grad: bool) -> None:
    tensor.requires_grad = requires_grad
    tensor.retain_grad = retain_grad
    gt = tensor > 0
    assert (gt.data == expected.data).all()
    assert gt.requires_grad is requires_grad
    assert gt.retain_grad is False


@pytest.mark.parametrize("tensor, axis, expected", [
    (Tensor([]), 0, Tensor([])),
    (Tensor([[1.0]]), 0, Tensor([1.0])),
    (Tensor([[1.0]]), 1, Tensor([1.0])),
    (Tensor([
        [1.0, 2.0, 3.0],
        [10.0, 20.0, 30.0],
    ]), 0, Tensor([11.0, 22.0, 33.0])),
    (Tensor([
        [1.0, 2.0, 3.0],
        [10.0, 20.0, 30.0],
    ]), 1, Tensor([6.0, 60.0])),
])
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("retain_grad", [True, False])
def test_sum(tensor: Tensor, axis: int, expected: Tensor, requires_grad: bool, retain_grad: bool) -> None:
    tensor.requires_grad = requires_grad
    tensor.retain_grad = retain_grad
    sum = tensor.sum(axis=axis)
    assert (sum.data == expected.data).all()
    assert sum.requires_grad is requires_grad
    assert sum.retain_grad is False


@pytest.mark.parametrize("tensor, expected", [
    (Tensor([]), Tensor([])),
    (Tensor([[1.0]]), Tensor([1.0])),
    (Tensor([
        [1.0, 2.0, 3.0],
        [10.0, 20.0, 30.0],
    ]), Tensor([11.0])),
])
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("retain_grad", [True, False])
def test_sum(tensor: Tensor, expected: Tensor, requires_grad: bool, retain_grad: bool) -> None:
    tensor.requires_grad = requires_grad
    tensor.retain_grad = retain_grad
    mean = tensor.mean()
    assert (mean.data == expected.data).all()
    assert mean.requires_grad is requires_grad
    assert mean.retain_grad is False


@pytest.mark.parametrize("tensor, expected", [
    (Tensor([]), Tensor([])),
    (Tensor([[1.0]]), Tensor([[1.0]])),
    (Tensor([[[1.0]]]), Tensor([[[1.0]]])),
    (Tensor([0, 1, 2, 3]), Tensor([0, 1, 2, 3])),
    (Tensor([[0, 1, 2, 3]]), Tensor([[0], [1], [2], [3]])),
])
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("retain_grad", [True, False])
def test_T(tensor: Tensor, expected: Tensor, requires_grad: bool, retain_grad: bool) -> None:
    tensor.requires_grad = requires_grad
    tensor.retain_grad = retain_grad
    t = tensor.T
    assert (t.data == expected.data).all()
    assert t.requires_grad is requires_grad
    assert t.retain_grad is False


@pytest.mark.parametrize("a, b, expected", [
    (Tensor([[4]]), Tensor([[2]]), Tensor([[8]])),
    (Tensor([
        [1, 0, 1],
        [2, 1, 1],
        [0, 1, 1],
        [1, 1, 2],
    ]), Tensor([
        [1, 2, 1],
        [2, 3, 1],
        [4, 2, 2],
    ]), Tensor([
        [5, 4, 3],
        [8, 9, 5],
        [6, 5, 3],
        [11, 9, 6,]
    ])),
])
@pytest.mark.parametrize("requires_grad_a", [True, False])
@pytest.mark.parametrize("requires_grad_b", [True, False])
@pytest.mark.parametrize("retain_grad", [True, False])
def test_matmul(a: Tensor, b: Tensor, expected: Tensor, requires_grad_a: bool, requires_grad_b: bool, retain_grad: bool) -> None:
    a.requires_grad = requires_grad_a
    a.retain_grad = retain_grad
    b.requires_grad = requires_grad_b
    b.retain_grad = retain_grad
    p = Tensor.matmul(a, b)
    assert (p.data == expected.data).all()
    assert p.requires_grad is (requires_grad_a or requires_grad_b)
    assert p.retain_grad is False


@pytest.mark.parametrize("shape, expected", [
    ((1,), Tensor([0.0])),
    ((2, 2), Tensor([[0.0, 0.0], [0.0, 0.0]])),
])
def test_zeros(shape: Shape, expected: Tensor) -> None:
    assert (Tensor.zeros(shape).data == expected.data).all()


@pytest.mark.parametrize("tensor, expected", [
    (Tensor([[4]]), Tensor([[0]])),
    (Tensor([
        [1, 0, 1],
        [2, 1, 1],
        [0, 1, 1],
        [1, 1, 2],
    ]), Tensor([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])),
])
def test_zeros_like(tensor: Tensor, expected: Tensor) -> None:
    assert (Tensor.zeros_like(tensor).data == expected.data).all()


def test_leaf() -> None:
    t = Tensor([[4]])
    assert t.is_leaf

    a = Tensor([[4]])
    t.inputs = (a, )
    assert not t.is_leaf


def test_backward() -> None:
    pass

