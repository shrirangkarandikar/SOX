import numpy as np
import math

# Taylor series for sine function
def create_sine(datapoints, dimensions=10):
    x_orr = np.random.uniform(-3, 3, datapoints)
    x_random = np.matrix([((-1)**(i+1)) * (x_orr**((2*i)-1)) for i in range(1, dimensions)]).T
    weights = np.matrix([1 / math.factorial((2*i)-1) for i in range(1, dimensions)]).T
    y_target = np.dot(x_random, weights)
    return x_orr, y_target

# Taylor series for cosine function
def create_cos(datapoints, dimensions=10):
    x_orr = np.random.uniform(-3, 3, datapoints)
    x_random = np.matrix([((-1)**i) * (x_orr**(2*i)) for i in range(dimensions)]).T
    weights = np.matrix([1 / math.factorial(2*i) for i in range(dimensions)]).T
    y_target = np.dot(x_random, weights)
    return x_orr, y_target

# Taylor series for exponential function (e^x)
def create_exp(datapoints, dimensions=10):
    x_orr = np.random.uniform(-3, 3, datapoints)
    x_random = np.matrix([(x_orr**i) for i in range(dimensions)]).T
    weights = np.matrix([1 / math.factorial(i) for i in range(dimensions)]).T
    y_target = np.dot(x_random, weights)
    return x_orr, y_target

# Geometric progression Taylor series (specific case where x < 1)
def create_geometric(datapoints, dimensions=10):
    x_orr = np.random.uniform(0, 0.9, datapoints)  # Ensure x < 1
    x_random = np.matrix([x_orr**i for i in range(1, dimensions + 1)]).T
    weights = np.matrix([1 for i in range(1, dimensions + 1)]).T  # 1 for geometric series
    y_target = np.dot(x_random, weights)
    return x_orr, y_target

# Alternating series Taylor expansion
def create_alternating_series(datapoints, dimensions=10):
    x_orr = np.random.uniform(-4, 4, datapoints)
    x_random = np.matrix([((-1)**(i+1)) * (x_orr**i) / i for i in range(1, dimensions + 1)]).T
    weights = np.ones((dimensions, 1))  # Taylor series predefined weights as 1
    y_target = np.dot(x_random, weights)
    return x_orr, y_target
