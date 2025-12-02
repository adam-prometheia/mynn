import numpy as np

from mynn.layers import Layer_Dense


def test_layer_dense_forward_backward_shapes():
    np.random.seed(0)
    layer = Layer_Dense(3, 4)
    X = np.random.randn(5, 3)

    layer.forward(X)
    assert layer.output.shape == (5, 4)

    upstream_grad = np.ones_like(layer.output)
    layer.backward(upstream_grad)

    assert layer.dweights.shape == (3, 4)
    assert layer.dbiases.shape == (1, 4)
    assert layer.dinputs.shape == (5, 3)


def test_layer_dense_gradient_consistency():
    np.random.seed(1)
    layer = Layer_Dense(2, 2)
    X = np.random.randn(4, 2)
    upstream_grad = np.random.randn(4, 2)

    layer.forward(X)
    layer.backward(upstream_grad)

    num_grad = (X.T @ upstream_grad)
    assert np.allclose(layer.dweights, num_grad, atol=1e-6)
