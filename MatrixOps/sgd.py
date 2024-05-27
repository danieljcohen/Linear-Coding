import numpy as np
from print import printMatrix

def compute_cost(X, y, theta):
    """
    Compute the cost for linear regression
    X: Input features, matrix of shape (m, n+1)
    y: Output/target variable, vector of shape (m, 1)
    theta: Parameters for regression, vector of shape (n+1, 1)
    
    Returns the cost
    """
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.dot(errors.T, errors)
    return cost[0, 0]

def sgd_step(X, y, theta, learning_rate):
    """
    Perform one step of SGD
    X: Input features, matrix of shape (m, n+1)
    y: Output/target variable, vector of shape (m, 1)
    theta: Parameters for regression, vector of shape (n+1, 1)
    learning_rate: Learning rate for the gradient descent step
    
    Returns the updated theta
    """
    m = len(y)
    for i in range(m):
        xi = X[i, :].reshape(1, -1)
        yi = y[i]
        prediction = xi.dot(theta)
        error = prediction - yi
        theta -= learning_rate * error * xi.T
    return theta

def stochastic_gradient_descent(X, y, theta, learning_rate, num_epochs):
    """
    Perform SGD for a number of epochs
    X: Input features, matrix of shape (m, n+1)
    y: Output/target variable, vector of shape (m, 1)
    theta: Parameters for regression, vector of shape (n+1, 1)
    learning_rate: Learning rate for the gradient descent step
    num_epochs: Number of epochs to run SGD
    
    Returns the updated theta and cost history
    """
    cost_history = []

    for epoch in range(num_epochs):
        theta = sgd_step(X, y, theta, learning_rate)
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        print(f"Epoch {epoch + 1}/{num_epochs}, Cost: {cost}")

    return theta, cost_history

if __name__ == '__main__':
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    X_b = np.c_[np.ones((100, 1)), X]

    theta = np.random.randn(2, 1)  # Random initialization
    learning_rate = 0.01
    num_epochs = 50

    theta, cost_history = stochastic_gradient_descent(X_b, y, theta, learning_rate, num_epochs)

    print("\nFinal theta values:")
    print(theta)

    import matplotlib.pyplot as plt

    plt.plot(range(1, num_epochs + 1), cost_history, label='Cost')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Cost vs. Epochs')
    plt.legend()
    plt.show()
