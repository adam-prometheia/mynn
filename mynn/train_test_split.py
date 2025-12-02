import numpy as np

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
