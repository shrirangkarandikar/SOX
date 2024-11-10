import numpy as np
import matplotlib.pyplot as plt
import math



def get_learning_rate_vector(iteration, initial_lrs, decay_rate=0.1):
    return initial_lrs / (1 + decay_rate * iteration)


def fit(dimensions, datapoints):
    max_iterations=1000000
    tolerance=0.00001
    x_orr = np.random.uniform(-3, 3, datapoints)
    x_random = np.matrix([(((-1)**(i)) * (x_orr**((2*i)))) for i in range(0, dimensions-1)]).T
    weights = np.matrix([1 / math.factorial((2*i)) for i in range(0, dimensions-1)]).T
    y_target = np.dot(x_random, weights)
    y = np.array(np.sin(x_orr))
    weights_n = (np.matrix([1 / math.factorial((2*i)) for i in range(0, dimensions-1)]) * 0.7).T
    initial_learning_rates = np.matrix([[0.000001],[0.00001],[0.0001]])
    error_mat=[]
    def p1(x, weights_n):
        return np.dot(x, weights_n)
    def calculate_errors(z):
        return np.square(z) / datapoints
    for i in range(max_iterations):
        y_mult = p1(x_random, weights_n)
        z = y_mult - y_target
        error = calculate_errors(z)
        error_mat.append(np.sum(error))
        learning_rates = get_learning_rate_vector(i, initial_learning_rates)
        grad_a1 = np.dot(z.T, x_random) * learning_rates / datapoints
        weights_n -= grad_a1.T
        if np.sum(error) < tolerance:
            break

    return error_mat, y_mult, x_orr, y_target, i


