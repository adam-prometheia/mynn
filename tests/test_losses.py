import numpy as np

from mynn.losses import Activation_Softmax_Loss_CategoricalCrossentropy


def test_softmax_crossentropy_probabilities_and_loss():
    logits = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 2.0]])
    y_true = np.array([0, 1])

    combined = Activation_Softmax_Loss_CategoricalCrossentropy()
    loss = combined.forward(logits, y_true)

    probs = combined.output
    assert np.allclose(np.sum(probs, axis=1), 1.0)

    manual = 0
    for sample_probs, label in zip(probs, y_true):
        manual += -np.log(sample_probs[label])
    manual /= len(y_true)

    assert np.isclose(loss, manual)

    combined.backward(probs, y_true)
    assert combined.dinputs.shape == logits.shape
