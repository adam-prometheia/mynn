import numpy as np

# Class for batch normalisation
class Layer_BatchNorm:
    def __init__(self):
        # Initialise a stabiliser to prevent division by zero
        self.epsilon = 1e-7
        self.gamma = 1.0 # Scale factor
        self.beta = 0.0 # Shift factor

    # Forward pass: normalise inputs
    def forward(self, inputs):
        # Store the input for the backward pass
        self.inputs = inputs
        
        # Calculate mean and standard deviation
        self.mean = np.mean(inputs, axis=0)
        self.std = np.std(inputs, axis=0)
        
        # Normalise the inputs
        self.normalised = (inputs - self.mean) / (self.std + self.epsilon)
        
        # Apply the learnable parameters gamma and beta
        self.output = self.gamma * self.normalised + self.beta

    # Backward pass: backpropagation for Batch Normalisation
    def backward(self, dvalues):
        # Number of samples in the batch
        N = dvalues.shape[0]

        # Derivatives of gamma and beta (used for parameter updates)
        self.dgamma = np.sum(dvalues * self.normalised, axis=0)
        self.dbeta = np.sum(dvalues, axis=0)

        # Derivative w.r.t. the normalised input
        dnormalised = dvalues * self.gamma

         # Backpropagate through normalisation
        dstd = np.sum(dnormalised * (self.inputs - self.mean) * -0.5 * (self.std + self.epsilon) ** (-3/2), axis=0)
        dmean = np.sum(dnormalised * -1 / (self.std + self.epsilon), axis=0) + dstd * np.mean(-2 * (self.inputs - self.mean), axis=0)
        
        # Derivative w.r.t. the input
        self.dinputs = dnormalised / (self.std + self.epsilon) + dstd * 2 * (self.inputs - self.mean) / N + dmean / N
        
        return self.dinputs
