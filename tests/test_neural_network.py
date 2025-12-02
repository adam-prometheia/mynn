import numpy as np

from mynn import Layer_Dense, Activation_ReLU, Neural_Network


def test_neural_network_trains_on_xor():
    np.random.seed(42)
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 1, 1, 0])

    layers = [
        Layer_Dense(2, 8),
        Activation_ReLU(),
        Layer_Dense(8, 2),
    ]
    model = Neural_Network(layers)

    model.train(
        X,
        y,
        epochs=1500,
        batch_size=4,
        patience=200,
        min_loss_delta=1e-5,
        use_batch_metrics=True,
        metric_updates=500,
    )

    loss, accuracy = model.evaluate(X, y)
    assert loss < 0.15
    assert accuracy == 1.0
