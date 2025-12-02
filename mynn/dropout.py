import numpy as np

# Class for dropping neurons
class Layer_Dropout:
    def __init__(self, rate):
        self.rate = rate  # Dropout rate (e.g., 0.2 means drop 20% of neurons)

    # Forward pass: Drop some layers
    def forward(self, inputs):
        self.inputs = inputs
        # Create a binary mask (1s where neurons stay active, 0s where dropped)
        self.binary_mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate)
        self.output = inputs * self.binary_mask

    # Backward pass: Pass dvalues only to non-dropped neurons
    def backward(self, dvalues):
        # Gradient flows only through neurons that weren’t dropped
        self.dinputs = dvalues * self.binary_mask
