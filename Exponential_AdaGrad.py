import numpy as np
import matplotlib.pyplot as plt
import math
import time

initial_weights = None
weight_updates = []
error_mat = []

def fit(dimensions, datapoints):
    x_orr = np.random.uniform(-3, 3, datapoints)
    x_random = np.matrix([((x_orr**((i)))) for i in range(0, dimensions-1)]).T
    weights = np.matrix([1 / math.factorial((i)) for i in range(0, dimensions-1)]).T
    y_target = np.dot(x_random, weights)
    y = np.array(np.sin(x_orr))
    weights_n = (np.matrix([1 / math.factorial((i)) for i in range(0, dimensions-1)]) * 0.7).T
    error_mat=[]
    def p1(x, weights_n):
        return np.dot(x, weights_n)
    
    def calculate_errors(z):
        return np.square(z)/datapoints

    
    learning_rate = 0.01  
    epsilon = 1e-8  

   
    gradient_sum = np.zeros(weights_n.shape)

    
    for i in range(100000):
        y_mult = p1(x_random, weights_n)
        z = y_mult - y_target
        error = calculate_errors(z)
        error_mat.append(np.sum(error))
        
        # Gradient calculation
        grad = np.dot(z.T, x_random).T / datapoints
        
        # Adagrad update: accumulate squared gradients
        gradient_sum += np.square(grad)
        
        # Update weights using Adagrad
        weights_n -= (learning_rate * grad) / (np.sqrt(gradient_sum) + epsilon)

        # Check if the error has fallen below the tolerance
        if np.sum(error) < 0.00001:
            break

    return error_mat, y_mult, x_orr, y_target, i

