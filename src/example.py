from typing import Iterator

import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix

from layers.linear import Linear
from layers.loss import BinaryCrossEntropyLoss
from layers.one_hot import OneHot
from layers.relu import ReLU
from layers.softmax import SoftMax
from optimizer import SGD
from tensor import Tensor
import random


if __name__ == "__main__":
    iris = datasets.load_iris()

    data = list(zip(iris["data"], iris["target"]))
    random.shuffle(data)
    train_set = data[0:130]
    train_x: list[list[float]] = []
    train_y: list[int] = []
    for x, y in train_set:
        train_x.append(x)
        train_y.append(y)

    test_set = data[130:]
    test_x: list[list[float]] = []
    test_y: list[int] = []
    for x, y in test_set:
        test_x.append(x)
        test_y.append(y)

    num_classes = max(iris["target"]) + 1

    linear_1 = Linear(len(iris["data"][0]), 3)
    linear_2 = Linear(3, num_classes)
    relu = ReLU()
    softmax = SoftMax()

    def forward(x: Tensor) -> Tensor:
        x = linear_1(x)
        x = linear_2(x)
        x = relu(x)
        x = softmax(x)
        return x

    oh = OneHot(num_classes)
    bcel = BinaryCrossEntropyLoss()

    lr = 0.01
    optimizer = SGD([linear_1.weight, linear_1.bias, linear_2.weight, linear_2.bias], lr)

    batch_size = 32

    def train_batches() -> Iterator[tuple[Tensor, Tensor]]:
        i = 0
        while i < len(train_x):
            yield Tensor(train_x[i:i + batch_size]), Tensor(train_y[i:i + batch_size])
            i += batch_size

    num_epochs = 100
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        for x, y in train_batches():
            probs = forward(x)

            loss = bcel(probs, oh(y))
            print("loss =", loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_x_ten = Tensor(test_x)
    test_y_ten = Tensor(test_y)
    r = forward(test_x_ten)
    predicted = np.argmax(r.data, axis=1)
    print("Test labels:", test_y_ten.data)
    print("Predicted labels:", predicted)

    print("Confusion matrix:")
    print(confusion_matrix(test_y_ten.data, predicted))
