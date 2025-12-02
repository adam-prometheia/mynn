#!/usr/bin/env python
"""Two-class spiral demo to visualise decision boundaries in a small MLP."""

import numpy as np

from mynn import Activation_ReLU, Layer_Dense, Neural_Network
from mynn.train_test_split import train_test_split


def build_model():
    """Define a slightly deeper MLP suitable for the spiral toy problem."""
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
        "Demo: classify a 2D spiral dataset (600 samples, two intertwined arms). "
        "Each point has radius r and angle theta with small Gaussian noise; labels alternate per arm."
    )

    # Two-class spiral dataset
    n_samples = 600
    n_classes = 2
    X = np.zeros((n_samples, 2), dtype=np.float64)
    y = np.zeros(n_samples, dtype=int)
    points_per_class = n_samples // n_classes

    for class_idx in range(n_classes):
        ix = range(class_idx * points_per_class, (class_idx + 1) * points_per_class)
        r = np.linspace(0.0, 1.0, points_per_class)  # radius
        t = np.linspace(
            class_idx * np.pi, (class_idx + 1) * np.pi, points_per_class
        ) + np.random.normal(0, 0.2, points_per_class)  # theta with noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = class_idx

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

    # Quick sanity check on a few held-out samples
    sample_idx = np.random.choice(len(X_test), size=5, replace=False)
    sample_pts = X_test[sample_idx]
    sample_labels = y_test[sample_idx]
    sample_probs = model.predict_probabilities(sample_pts)
    sample_preds = model.predict(sample_pts)
    print("Sample points:\n", np.round(sample_pts, 3))
    print("True labels:    ", sample_labels)
    print("Pred labels:    ", sample_preds)
    print("Pred probs:\n", np.round(sample_probs, 3))


if __name__ == "__main__":
    main()
