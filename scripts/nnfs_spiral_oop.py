"""OOP-style NNFS spiral demo demonstrating layers/activations as classes."""

import numpy as np
import copy
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt


# Initialise NNFS settings (e.g., random seed, float32 precision)
nnfs.init()

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

# Class for batch normalisation
class Layer_BatchNorm:
    # Initialise a stabiliser to prevent division by zero
    def __init__(self):
        self.epsilon = 1e-7

    # Forward pass: normalise inputs
    def forward(self, inputs):
        self.inputs = inputs
        self.mean = np.mean(inputs, axis=0)
        self.std = np.std(inputs, axis=0)
        self.output = (inputs - self.mean) / (self.std + self.epsilon)

    # Backward pass: simplified backpropagation
    def backward(self, dvalues):
        # Simplified backprop assuming fixed mean/std (no learnable gamma/beta)
        N = dvalues.shape[0]
        self.dinputs = (
            (dvalues - np.mean(dvalues, axis=0)) /
            (self.std + self.epsilon)
        )

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

# Class to implement ADAM - Momementum, adaptive learning rates, and bias correction
class Optimiser_Adam:
    def __init__(self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        # Initialise initial parameters
        self.learning_rate = learning_rate
        self.beta_1 = beta_1  # Momentum decay
        self.beta_2 = beta_2  # Cache decay
        self.epsilon = epsilon
        self.iteration = 0

    # Update parameters
    def update_parameters(self, layer):
        # Skip layers without any weights to train
        if not hasattr(layer, 'weights'):
            return

        # Initialise moving averages for the layer
        if not hasattr(layer, 'm_w'):
            layer.m_w = np.zeros_like(layer.weights) # Weights momentum
            layer.v_w = np.zeros_like(layer.weights) # Weights cache
            layer.m_b = np.zeros_like(layer.biases) # Biases momentum
            layer.v_b = np.zeros_like(layer.biases) # Biases cache

        # Increment step counter
        self.iteration += 1

        # -- Momentum updates (first moment) --
        layer.m_w = self.beta_1 * layer.m_w + (1 - self.beta_1) * layer.dweights
        layer.m_b = self.beta_1 * layer.m_b + (1 - self.beta_1) * layer.dbiases

        # -- RMSProp updates (second moment) --
        layer.v_w = self.beta_2 * layer.v_w + (1 - self.beta_2) * (layer.dweights ** 2)
        layer.v_b = self.beta_2 * layer.v_b + (1 - self.beta_2) * (layer.dbiases ** 2)

        # -- Bias correction --
        m_w_corr = layer.m_w / (1 - self.beta_1 ** self.iteration)
        m_b_corr = layer.m_b / (1 - self.beta_1 ** self.iteration)
        v_w_corr = layer.v_w / (1 - self.beta_2 ** self.iteration)
        v_b_corr = layer.v_b / (1 - self.beta_2 ** self.iteration)

        # -- Final parameter update --
        layer.weights -= self.learning_rate * m_w_corr / (np.sqrt(v_w_corr) + self.epsilon)
        layer.biases  -= self.learning_rate * m_b_corr / (np.sqrt(v_b_corr) + self.epsilon)

# Class to tie layers together
class Neural_Network:
    def __init__(self, layers, loss_function=None, optimiser=None):
        assert layers and isinstance(layers, list), "Layers must be a non-empty list."
        self.layers = layers
        self.loss_function = loss_function or Activation_Softmax_Loss_CategoricalCrossentropy()
        self.optimiser = optimiser or Optimiser_Adam(learning_rate=0.001)

        assert callable(getattr(self.loss_function, 'forward', None)), "Loss function must implement 'forward'"
        assert callable(getattr(self.optimiser, 'update_parameters', None)), "Optimizer must implement 'update_parameters'"
        
        # Initialise accuracy/loss tracking
        self.loss_history = []
        self.accuracy_history = []

    # Forward pass through all layers
    def forward(self, X):
        input_data = X
        for layer in self.layers:
            layer.forward(input_data)
            input_data = layer.output
        return input_data

    # Backward pass through all layers
    def backward(self):
        dvalues = self.loss_function.dinputs
        for layer in reversed(self.layers):
            layer.backward(dvalues)
            dvalues = layer.dinputs

    # Update trainable parameters
    def update_parameters(self):
        for layer in self.layers:
            self.optimiser.update_parameters(layer)

    # Prediction for inputs
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

    # Evaluate model performance
    def evaluate(self, X, y):
        predictions = self.forward(X)
        loss = self.loss_function.forward(predictions, y)
        accuracy = self._accuracy(predictions, y)
        return loss, accuracy

    # Compute accuracy
    def _accuracy(self, predictions, y_true):
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        pred_labels = np.argmax(predictions, axis=1)
        return np.mean(pred_labels == y_true)

    # Train model
    def train(self, X, y, epochs=3000, batch_size=64, patience=100, min_loss_delta=1e-6, use_batch_metrics=False):

        # Return mini batches based on batch size
        def create_mini_batches(X, y, batch_size):
            mini_batches = []
            for i in range(0, len(X), batch_size):
                mini_batches.append((X[i:i + batch_size], y[i:i + batch_size]))
            return mini_batches

        # Initialise best metric save variables
        best_loss = float('inf')
        best_accuracy = 0
        best_epoch = -1
        best_weights = []
        best_biases = []

        # Initialise early stop counter
        early_stop_counter = 0

        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Initialise mini batches
            mini_batches = create_mini_batches(X_shuffled, y_shuffled, batch_size)

            # Initialise last batch metrics
            last_batch_loss = 0
            last_batch_accuracy = 0

            for X_batch, y_batch in mini_batches:

                # Perform forward, loss, backward, and update for each mini batch
                output = self.forward(X_batch)
                loss = self.loss_function.forward(output, y_batch)
                self.loss_function.backward(self.loss_function.output, y_batch)
                self.backward()
                self.update_parameters()

                # Calculate mini batch metrics
                last_batch_loss = loss
                last_batch_accuracy = self._accuracy(output, y_batch)

            # Choose metrics source
            if use_batch_metrics:
                tracked_loss = last_batch_loss
                tracked_accuracy = last_batch_accuracy
            else:
                tracked_loss, tracked_accuracy = self.evaluate(X, y)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}: loss={tracked_loss:.4f}, accuracy={tracked_accuracy:.4f}")

                # Append metrics to history
                self.loss_history.append(tracked_loss)
                self.accuracy_history.append(tracked_accuracy)

            # Early stopping check
            if tracked_loss < best_loss - min_loss_delta:
                best_loss = tracked_loss
                best_accuracy = tracked_accuracy
                best_epoch = epoch

                # Save the new best weights and biases
                best_weights = [copy.deepcopy(layer.weights) for layer in self.layers if hasattr(layer, 'weights')]
                best_biases = [copy.deepcopy(layer.biases) for layer in self.layers if hasattr(layer, 'biases')]

                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. No significant improvement for {patience} epochs.")
                break

        # Restore best weights and biases from the epoch with the lowest loss
        print(f"Restoring best weights from epoch {best_epoch+1} with loss={best_loss:.4f} and accuracy={best_accuracy:.4f}")
        dense_layers = [layer for layer in self.layers if hasattr(layer, 'weights')]
        for i, layer in enumerate(dense_layers):
            layer.weights = best_weights[i]
            layer.biases = best_biases[i]

    # Plot the training loss and accuracy across epochs
    def plot_training(self):
        epochs = range(len(self.loss_history))
        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.loss_history, label='Loss', color='tomato')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.accuracy_history, label='Accuracy', color='seagreen')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

# Function to split data
def train_test_split(X, y, test_size=0.2, random_state=None):
    assert len(X) == len(y), "X and y must be the same length."

    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    # Generate a shuffled set of indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Shuffle X and y using the same indices
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Determine the split index
    test_count = int(len(X) * test_size)

    # Slice into training and testing sets
    X_test = X_shuffled[:test_count]
    y_test = y_shuffled[:test_count]
    X_train = X_shuffled[test_count:]
    y_train = y_shuffled[test_count:]

    return X_train, X_test, y_train, y_test

# Load sample dataset
sample_size = 1000
classes_size = 5
X, y = spiral_data(samples=sample_size, classes=classes_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create network
model = Neural_Network(
    layers=[
        Layer_Dense(2, 128),
        Layer_BatchNorm(),
        Leaky_ReLU(),
        Layer_Dense(128, 128),
        Layer_BatchNorm(),
        Leaky_ReLU(),
        Layer_Dense(128, 96),
        Layer_BatchNorm(),
        Leaky_ReLU(),
        Layer_Dense(96, classes_size)
    ],
    loss_function=Activation_Softmax_Loss_CategoricalCrossentropy(),
    optimiser=Optimiser_Adam(learning_rate=0.0005)
)

# Train network
model.train(
    X_train,
    y_train,
    epochs=10000,
    batch_size=1000,
    patience=300,
    min_loss_delta=1e-4,
    use_batch_metrics=False
    )

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
