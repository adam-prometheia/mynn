import numpy as np

# ReLU activation function
class Activation_ReLU:
    # Forward pass: apply ReLU activation
    def forward(self, inputs):
        self.inputs = inputs  # Save input for backprop
        self.output = np.maximum(0, inputs)  # Replace negative values with 0

    # Backward pass: gradient flows only where input > 0
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0  # Zero out gradients where ReLU was inactive

# Leaky ReLU activation function
class Leaky_ReLU:
    def __init__(self, alpha=0.01):
        # Slope for negative values
        self.alpha = alpha
    
    # Forward pass: apply Leaky ReLU activation
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)

    # Backward pass: gradient flows where input > 0 or is multiplied by alpha
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] *= self.alpha

# Softmax activation function
class Activation_Softmax:
    def forward(self, inputs):
        # Stabilise input values by subtracting max per sample
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalise to get probabilities
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
