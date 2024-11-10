import numpy as np
import matplotlib.pyplot as plt
import math

def fit(dimensions, datapoints):
    x_orr = np.random.uniform(-3, 3, datapoints)
    x_random = np.matrix([(((-1)**(i))*(x_orr**((2*i)))) for i in range(0, dimensions-1)]).T    
    weights = np.matrix([1/math.factorial((2*i)) for i in range(0, dimensions-1)]).T
    y_target = np.dot(x_random, weights)
    y = np.array(np.sin(x_orr))   
    weights_n = (np.matrix([1/math.factorial((2*i)) for i in range(0, dimensions-1)])*(0.7)).T
    def p1(x, weights_n):
        return np.dot(x, weights_n)
    
    def calculate_errors(z):
        return np.square(z) / datapoints
    
    learning_rate = 0.001  
    decay_rate = 0.9  
    epsilon = 1e-8  
    v = np.zeros(weights_n.shape) 
    error_mat=[] 
    
    for i in range(100000):
        y_mult = p1(x_random, weights_n)
        z = y_mult - y_target
        error = calculate_errors(z) 
        error_mat.append(np.sum(error))
        grad = np.dot(z.T, x_random).T / datapoints              
        
        v = decay_rate * v + (1 - decay_rate) * np.square(grad)       
        weights_n -= (learning_rate * grad) / (np.sqrt(v) + epsilon)
        
        if np.sum(error) < 0.00001:
            break
    return error_mat, y_mult, x_orr, y_target, i

