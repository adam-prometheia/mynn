"""Functional-style NNFS spiral demo with explicit layers/gradients.

Trains a multi-class classifier on the nnfs spiral dataset and renders decision
boundaries. Intended as an educational script; mirrors the book-style approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import copy
import nnfs
from nnfs.datasets import spiral_data

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
    # Forward pass: run softmax, then compute loss
    def forward(self, inputs, y_true):
        self.activation = Activation_Softmax()
        self.activation.forward(inputs)
        self.output = self.activation.output  # Softmax output probabilities

        self.loss = Loss_CategoricalCrossentropy()
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


# Load sample dataset
sample_size = 1000
classes_size = 5
X, y = spiral_data(samples=sample_size, classes=classes_size)

# Create a grid of points to evaluate
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Flatten the grid to pass through the network
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Create network layers
dense1 = Layer_Dense(2, 256)  # Input layer --> hidden layer 1 (2 inputs to 256 neurons)
activation1 = Activation_ReLU()  # ReLU activation

dense2 = Layer_Dense(256, 128)  # Hidden layer 1 --> hidden layer 2 (256 neurons to 128 neurons)
activation2 = Activation_ReLU()

dense3 = Layer_Dense(128, classes_size)  # Hidden layer --> output layer (128 neurons to number of classes)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Training loop
epochs = 1000
batch_size = 32

# Create mini batches for gradient descent
def create_mini_batches(X, y, batch_size):
    mini_batches = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]
        mini_batches.append((X_batch, y_batch))
    return mini_batches

# Dynamic learning rate
initial_learning_rate = 0.03
decay = 2e-3
start_decay_epoch = 0
stop_decay_epoch = epochs

def learning_rate_decay(epoch):
    if epoch < start_decay_epoch:
        return initial_learning_rate
    elif epoch < stop_decay_epoch:
        learning_rate = initial_learning_rate / (1 + decay * epoch)
    else:
        learning_rate = initial_learning_rate / (1 + decay * stop_decay_epoch)
    return learning_rate

# Early stopping
patience = 100  # Number of epochs to wait before stopping if no improvement
early_stop_counter = 0  # Counter to track the number of epochs without improvement
min_loss_delta = 1e-4 # Minimum loss improvement to reset the counter

# For plotting loss and accuracy across epochs
losses = []
accuracies = []

# For animating results
prediction_frames = []
epoch_checkpoints = []

def predict_grid():
    dense1.forward(grid_points)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    softmax = Activation_Softmax()
    softmax.forward(dense3.output)
    return np.argmax(softmax.output, axis=1).reshape(xx.shape)

# Storing best values
best_loss = float('inf')
best_accuracy = 0
best_weights = None
best_biases = None

smoothed_loss = None
smoothed_acc = None
ema_beta = 0.9  # smoothing factor

for epoch in range(epochs):

    # Shuffle the data for mini-batches
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Create mini-batches
    mini_batches = create_mini_batches(X_shuffled, y_shuffled, batch_size)
    batch_losses = []
    batch_accs = []
    
    # Loop through mini-batches
    for X_batch, y_batch in mini_batches:
        # FORWARD PASS for the mini-batch
        dense1.forward(X_batch)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        dense3.forward(activation2.output)
        loss = loss_activation.forward(dense3.output, y_batch)

        # Calculate accuracy for the mini-batch
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y_batch.shape) == 2:
            y_batch = np.argmax(y_batch, axis=1)
        accuracy = np.mean(predictions == y_batch)

        batch_losses.append(loss)
        batch_accs.append(accuracy)

        # Backward pass for the mini-batch
        loss_activation.backward(loss_activation.output, y_batch)
        dense3.backward(loss_activation.dinputs)
        activation2.backward(dense3.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Apply learning rate decay
        learning_rate = learning_rate_decay(epoch)
        
        # UPDATE WEIGHTS AND BIASES
        dense1.weights -= learning_rate * dense1.dweights
        dense1.biases  -= learning_rate * dense1.dbiases
        dense2.weights -= learning_rate * dense2.dweights
        dense2.biases  -= learning_rate * dense2.dbiases
        dense3.weights -= learning_rate * dense3.dweights
        dense3.biases  -= learning_rate * dense3.dbiases

    epoch_loss = float(np.mean(batch_losses))
    epoch_acc = float(np.mean(batch_accs))
    smoothed_loss = epoch_loss if smoothed_loss is None else ema_beta * smoothed_loss + (1 - ema_beta) * epoch_loss
    smoothed_acc = epoch_acc if smoothed_acc is None else ema_beta * smoothed_acc + (1 - ema_beta) * epoch_acc

    # Calculate and store loss and accuracy for plotting
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}: loss={epoch_loss:.4f} (ema {smoothed_loss:.4f}), accuracy={epoch_acc:.4f} (ema {smoothed_acc:.4f})")
        prediction_frames.append(predict_grid())
        epoch_checkpoints.append(epoch)
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

    # Save best weights and biases based on lowest loss (or highest accuracy)
    if epoch_loss < best_loss - min_loss_delta:
        best_loss = epoch_loss
        best_accuracy = epoch_acc
        best_epoch = epoch

        best_weights = {
            'dense1': copy.deepcopy(dense1.weights),
            'dense2': copy.deepcopy(dense2.weights),
            'dense3': copy.deepcopy(dense3.weights)
        }

        best_biases = {
            'dense1': copy.deepcopy(dense1.biases),
            'dense2': copy.deepcopy(dense2.biases),
            'dense3': copy.deepcopy(dense3.biases)
        }

        # Reset early stop counter
        early_stop_counter = 0
    else:
        # Increment early stop counter
        early_stop_counter += 1

    # Check for early stopping
    if early_stop_counter >= patience:
        print(f"Early stopping at epoch {epoch}. No real improvement in the last {patience} epochs.")
        break

print(f"Sample Size: {sample_size}")
print(f"Number of Classes: {classes_size}")
print(f"Batch Size: {batch_size}")
print(f"Initial Learning Rate: {initial_learning_rate}")
print(f"Rate of decay per epoch: {decay}")
print(f"Best Accuracy: {best_accuracy}")
print(f"Best Loss: {best_loss}")
print(f"Best epoch at: {best_epoch}")

# Restore best weights and biases
dense1.weights = best_weights['dense1']
dense1.biases = best_biases['dense1']
dense2.weights = best_weights['dense2']
dense2.biases = best_biases['dense2']
dense3.weights = best_weights['dense3']
dense3.biases = best_biases['dense3']

# Plot loss and accuracy over epochs
plt.plot(losses, label="Loss")
plt.plot(accuracies, label="Accuracy")
plt.legend()
plt.show()

# Forward pass through the trained network
dense1.forward(grid_points)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
softmax = Activation_Softmax()
softmax.forward(dense3.output)

# Predict classes and reshape to match the meshgrid
predictions = np.argmax(softmax.output, axis=1)
predictions = predictions.reshape(xx.shape)

# Plot decision boundaries
plt.contourf(xx, yy, predictions, cmap=plt.cm.Spectral, alpha=0.6)

# Plot original spiral data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')

plt.title("Spiral Data Decision Boundary with Scatter Points")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Plot decision boundaries
plt.contourf(xx, yy, predictions, cmap=plt.cm.Spectral, alpha=0.6)

plt.title("Spiral Data Decision Boundary without Scatter Points")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Animate the change in decision boundary over epochs (single animation, all frames)
fig1, ax1 = plt.subplots()

def update_with_scatter(i):
    ax1.clear()
    ax1.contourf(xx, yy, prediction_frames[i], cmap=plt.cm.Spectral, alpha=0.6)
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    ax1.set_title(f"Epoch {epoch_checkpoints[i]}\nAcc: {accuracies[i]:.2%} | Loss: {losses[i]:.4f}")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")

anim = FuncAnimation(fig1, update_with_scatter, frames=len(prediction_frames), interval=200)
plt.show()

# Save the animation as a .mp4 (all frames captured)
writer = FFMpegWriter(fps=10)
filename = "decision_boundary.mp4"
print("Saving animation...")
anim.save(filename, writer=writer)
print(f"Animation saved to: {filename}")
