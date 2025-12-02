import numpy as np
import copy
import matplotlib.pyplot as plt
from .optimisers import Optimiser_Adam
from .losses import Activation_Softmax_Loss_CategoricalCrossentropy

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

    # Predicition for softmax probabilities
    def predict_probabilities(self, X):
        logits = self.forward(X)
        self.loss_function.activation.forward(logits)
        return self.loss_function.activation.output

    # Prediction for inputs (returns predicted class)
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
    def train(self, X, y, epochs=3000, batch_size=64, patience=100, min_loss_delta=1e-6, use_batch_metrics=False, metric_updates=100):

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

            if epoch % metric_updates == 0:
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
