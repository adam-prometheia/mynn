#!/usr/bin/env python
"""XOR demo with noise + train/test split to stress the network a bit more."""

import numpy as np

from mynn import Activation_ReLU, Layer_Dense, Neural_Network
from mynn.train_test_split import train_test_split


def build_model():
    """Define a slightly deeper MLP for noisy XOR classification."""
    layers = [
        Layer_Dense(2, 16),
        Activation_ReLU(),
        Layer_Dense(16, 16),
        Activation_ReLU(),
        Layer_Dense(16, 2),
    ]
    return Neural_Network(layers)


def main():
    np.random.seed(42)
    print(
        "Demo: classify a noisy XOR dataset (400 samples). "
        "Each sample is two binary inputs with Gaussian noise; label = 1 when bits differ."
    )

    # Noisy XOR: sample 2-bit inputs, add Gaussian noise, labels via xor
    n_samples = 400
    bits = np.random.randint(0, 2, size=(n_samples, 2))
    noise = np.random.normal(loc=0.0, scale=0.15, size=bits.shape)
    X = bits.astype(np.float64) + noise
    y = (bits[:, 0] ^ bits[:, 1]).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = build_model()
    model.train(
        X_train,
        y_train,
        epochs=3000,
        batch_size=32,
        patience=400,
        min_loss_delta=5e-6,
        use_batch_metrics=False,
        metric_updates=300,
    )

    train_loss, train_acc = model.evaluate(X_train, y_train)
    test_loss, test_acc = model.evaluate(X_test, y_test)

    print(f"Train loss/acc: {train_loss:.4f} / {train_acc:.4f}")
    print(f"Test  loss/acc: {test_loss:.4f} / {test_acc:.4f}")

    # Show behaviour on the canonical XOR corners (no noise)
    canonical = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    probs = model.predict_probabilities(canonical)
    preds = model.predict(canonical)
    print("Canonical probabilities:\n", np.round(probs, 3))
    print("Canonical predictions:", preds)


if __name__ == "__main__":
    main()
