import numpy as np

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

        # Initialise momentum and cache for gamma and beta (if they exist)
        if hasattr(layer, 'gamma'):
            layer.m_gamma = np.zeros_like(layer.gamma)
            layer.v_gamma = np.zeros_like(layer.gamma)
        if hasattr(layer, 'beta'):
            layer.m_beta = np.zeros_like(layer.beta)
            layer.v_beta = np.zeros_like(layer.beta)

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

        # -- Handle gamma and beta updates for Batch Normalization layers --
        if hasattr(layer, 'gamma'):
            # Momentum update for gamma
            layer.m_gamma = self.beta_1 * layer.m_gamma + (1 - self.beta_1) * layer.dgamma
            # RMSProp update for gamma
            layer.v_gamma = self.beta_2 * layer.v_gamma + (1 - self.beta_2) * (layer.dgamma ** 2)

            # Bias correction for gamma
            m_gamma_corr = layer.m_gamma / (1 - self.beta_1 ** self.iteration)
            v_gamma_corr = layer.v_gamma / (1 - self.beta_2 ** self.iteration)

            # Final update for gamma
            layer.gamma -= self.learning_rate * m_gamma_corr / (np.sqrt(v_gamma_corr) + self.epsilon)

        if hasattr(layer, 'beta'):
            # Momentum update for beta
            layer.m_beta = self.beta_1 * layer.m_beta + (1 - self.beta_1) * layer.dbeta
            # RMSProp update for beta
            layer.v_beta = self.beta_2 * layer.v_beta + (1 - self.beta_2) * (layer.dbeta ** 2)

            # Bias correction for beta
            m_beta_corr = layer.m_beta / (1 - self.beta_1 ** self.iteration)
            v_beta_corr = layer.v_beta / (1 - self.beta_2 ** self.iteration)

            # Final update for beta
            layer.beta -= self.learning_rate * m_beta_corr / (np.sqrt(v_beta_corr) + self.epsilon)
