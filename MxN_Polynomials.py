import numpy as np


def create_data(polynomial_length,polynomial_count,polynomial_outputs):
    '''
    x_random : Input of the polynomials, shape(x_count * data_points)
    weights : Weights of the final polynomial, shape(x_count * y_Counts)
    p(x) : Computes the final polynomial
    y : Final Polynomial, shape(y*data_points)
    '''
    x_random = np.random.uniform(-1, 1, (polynomial_length, polynomial_count))  
    x_random=x_random.T
    weights = np.random.uniform(0, 10, (polynomial_length, polynomial_outputs))  
    
    def p(x):
        return np.dot(x, weights)
    y = p(x_random)
    return x_random,y

def update(polynomial_length,polynomial_outputs,x_random,y):
    '''
    
    '''
    learning_rate = 0.01
    weights_n = np.random.uniform(0, 10, (polynomial_length, polynomial_outputs))
    def p1(x):
        return np.dot(x, weights_n)
    def calculate_errors(z):
        return ((z)**2)/polynomial_length    
    for i in range(10000):
        y_mult = p1(x_random)        
        z = y_mult-y 
        error = calculate_errors(z)
        grad_a1 = np.dot(x_random.T,z)*learning_rate/polynomial_length        
        weights_n = weights_n - grad_a1        
        if (error <= 1e-20).all():
            break
    return weights_n,error


