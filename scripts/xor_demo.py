#!/usr/bin/env python
"""Minimal XOR training demo for the mynn package."""

import numpy as np

from mynn import Activation_ReLU, Layer_Dense, Neural_Network


def build_model():
    layers = [
        Layer_Dense(2, 8),
        Activation_ReLU(),
        Layer_Dense(8, 2),
    ]
    return Neural_Network(layers)


def main():
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

    model = build_model()
    model.train(
        X,
        y,
        epochs=2000,
        batch_size=4,
        patience=300,
        min_loss_delta=1e-5,
        use_batch_metrics=True,
        metric_updates=500,
    )

    probs = model.predict_probabilities(X)
    preds = model.predict(X)
    print("Probabilities:\n", np.round(probs, 3))
    print("Predictions:", preds)


if __name__ == "__main__":
    main()
