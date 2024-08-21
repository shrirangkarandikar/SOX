import numpy as np


def create_data(polynomial_length,polynomial_count):
    """
    Create random polynomial data.
    
    Args:
    - degree (int): Degree of the polynomial.
    - data_points (int): Number of data points to generate.

    Returns:
    - x_random (np.ndarray): Random input data for the polynomial.
    - y (np.ndarray): Output data after applying the polynomial.
    - num (list): List of indices for data points.
    """
    a = np.random.uniform(0, 30, polynomial_length).reshape(1, polynomial_length)
    def p(x):
        return np.sum(np.matmul(a, x), axis=0)
    x_random = np.random.uniform(-1, 1, (polynomial_length, polynomial_count))
    y = p(x_random)
    return x_random, y

def update(polynomial_length,polynomial_count,x_random,y):    
    learning_rate = 0.01
    a_new = np.random.randint(-100, 300, polynomial_length).reshape(1, polynomial_length)
    def p_new(a, x):
        return np.sum(np.matmul(a, x), axis=0)
    def error_calc(y):
        return np.sum(y)   
    mat = np.zeros((polynomial_length+1,polynomial_count))
    mat[1:polynomial_length+1,:] = x_random
    for i in range(10000):        
        y_final = p_new(a_new,x_random)        
        mat[0:1,:] = y_final-y.reshape(1,polynomial_count)
        mat1 = mat[0:1,:]*mat        
        error = error_calc(mat1[0:1,:])
        if error<=1e-20:
            break
        grad_a1=(np.sum(mat1[1:polynomial_length+1,:],axis=1))*learning_rate
        a_new=a_new-grad_a1
        
    return a_new,error