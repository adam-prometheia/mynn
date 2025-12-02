import numpy as np

from mynn.train_test_split import train_test_split


def test_train_test_split_reproducible():
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20)

    split_one = train_test_split(X, y, test_size=0.25, random_state=123)
    split_two = train_test_split(X, y, test_size=0.25, random_state=123)

    assert all(np.array_equal(a, b) for a, b in zip(split_one, split_two))

    X_train, X_test, y_train, y_test = split_one
    assert len(X_test) == 5
    assert len(X_train) == 15
    assert len(y_test) == 5
    assert len(y_train) == 15
