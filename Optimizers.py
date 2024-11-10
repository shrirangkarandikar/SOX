import numpy as np

def Decay(x_random, y_target, weights_n, learning_rate, iterations):
    datapoints = len(y_target)
    error = []

    for i in range(iterations):
        y_mult = np.dot(x_random, weights_n)  # Ensure shapes align
        z = y_mult - y_target  # This should keep the shape as (100, 1)
        
        # Calculate the gradient
        grad_a1 = np.dot(x_random.T, z) * learning_rate / (datapoints*10)  # Shape should be (5, 1)
        
        weights_n -= grad_a1  # Update weights
        loss = np.mean(z**2)
        error.append(loss)

    return weights_n, error

def RMSProp(x_random, y_target, weights_n, learning_rate, iterations, beta=0.9, epsilon=1e-8):
    datapoints = len(y_target)
    error = []
    cache = np.zeros_like(weights_n)

    for i in range(iterations):
        y_mult = np.dot(x_random, weights_n)
        z = y_mult - y_target  # This should keep the shape as (100, 1)
        
        grad_a1 = np.dot(x_random.T, z) * learning_rate / datapoints
        cache = beta * cache + (1 - beta) * grad_a1**2
        weights_n -= (learning_rate * grad_a1) / (np.sqrt(cache) + epsilon)
        loss = np.mean(z**2)
        error.append(loss)

    return weights_n, error

def Adam(x_random, y_target, weights_n, learning_rate, iterations, beta1=0.9, beta2=0.999, epsilon=1e-8):
    datapoints = len(y_target)
    error = []
    m = np.zeros_like(weights_n)
    v = np.zeros_like(weights_n)

    for i in range(iterations):
        y_mult = np.dot(x_random, weights_n)
        z = y_mult - y_target  # This should keep the shape as (100, 1)
        
        grad_a1 = np.dot(x_random.T, z) * learning_rate / datapoints

        m = beta1 * m + (1 - beta1) * grad_a1
        v = beta2 * v + (1 - beta2) * grad_a1**2

        m_hat = m / (1 - beta1**(i + 1))
        v_hat = v / (1 - beta2**(i + 1))

        weights_n -= (learning_rate * m_hat) / (np.sqrt(v_hat) + epsilon)
        loss = np.mean(z**2)
        error.append(loss)

    return weights_n, error

def Adagrad(x_random, y_target, weights_n, learning_rate, iterations, epsilon=1e-8):
    datapoints = len(y_target)
    error = []
    cache = np.zeros_like(weights_n)

    for i in range(iterations):
        y_mult = np.dot(x_random, weights_n)
        z = y_mult - y_target  # This should keep the shape as (100, 1)

        grad_a1 = np.dot(x_random.T, z) * learning_rate / datapoints
        cache += grad_a1**2
        weights_n -= (learning_rate * grad_a1) / (np.sqrt(cache) + epsilon)
        loss = np.mean(z**2)
        error.append(loss)

    return weights_n, error
