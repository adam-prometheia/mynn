import numpy as np
from .activations import Activation_Softmax

# Base loss class
class Loss:
    # Calculates mean loss over all samples
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


# Categorical Crossentropy loss function
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # If labels are sparse integers
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # If labels are one-hot encoded
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Compute negative log likelihoods
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# Combined Softmax activation + Crossentropy loss
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        # Initialise activation + loss to maintain constant values in backpropagation
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    
    # Forward pass: run softmax, then compute loss
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output  # Softmax output probabilities
        return self.loss.calculate(self.output, y_true)

    # Backward pass: simplified gradient for softmax + crossentropy
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # If labels are one-hot, convert to sparse
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy softmax outputs and subtract 1 at the true class index
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1

        # Normalise the gradient over the batch
        self.dinputs = self.dinputs / samples

