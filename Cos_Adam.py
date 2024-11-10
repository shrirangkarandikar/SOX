import numpy as np
import matplotlib.pyplot as plt
import math
import time

initial_weights = None
weight_updates = []
error_mat = []

def fit(dimensions,datapoints):
    x_orr = np.random.uniform(-3, 3, datapoints)
    x_random = np.matrix([(((-1)**(i))*(x_orr**((2*i)))) for i in range(0, dimensions-1)]).T    
    weights = np.matrix([1/math.factorial((2*i)) for i in range(0, dimensions-1)]).T
    y_target = np.dot(x_random, weights)
    y = np.array(np.sin(x_orr))   
    weights_n = (np.matrix([1/math.factorial((2*i)) for i in range(0, dimensions-1)])*(0.7)).T
    error_mat=[]
    def p1(x, weights_n):
        return np.dot(x, weights_n)
    
    def calculate_errors(z):
        return np.square(z)/datapoints   
    learning_rate = 0.001  
    beta1 = 0.9  
    beta2 = 0.999  
    epsilon = 1e-8      
    m = np.zeros(weights_n.shape)  
    v = np.zeros(weights_n.shape)  
    
    for i in range (100000):
        y_mult = p1(x_random, weights_n)
        z = y_mult - y_target
        error = calculate_errors(z) 
        error_mat.append(np.sum(error))
        grad = np.dot(z.T, x_random).T / datapoints              
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * np.square(grad)       
        m_hat = m / (1 - beta1**(i+1))
        v_hat = v / (1 - beta2**(i+1))      
        weights_n -= (learning_rate * m_hat) / (np.sqrt(v_hat) + epsilon)
        if np.sum(error) < 0.00001:
            break
    return error_mat, y_mult, x_orr, y_target, i

