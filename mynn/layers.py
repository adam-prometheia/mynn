import numpy as np

# Dense (fully connected) layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialise weights with small random values and biases with zeros
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs) # He initiliasation
        self.biases = np.zeros((1, n_neurons))

    # Forward pass: compute output values
    def forward(self, inputs):
        self.inputs = inputs  # Save input for backprop
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass: calculate gradients for weights, biases, and inputs
    def backward(self, dvalues):
        # Gradient w.r.t. weights
        self.dweights = np.dot(self.inputs.T, dvalues)
        # Gradient w.r.t. biases
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient w.r.t. inputs to this layer (to pass backward)
        self.dinputs = np.dot(dvalues, self.weights.T)
