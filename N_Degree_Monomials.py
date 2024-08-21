import numpy as np

def create_data(polynomial_length,polynomial_count):    
    a = np.random.uniform(0, 30, polynomial_length).reshape(1, polynomial_length)
    def p(x):
        return np.sum(np.matmul(a, x), axis=0)
    x_random = np.random.uniform(-1, 1, polynomial_count)
    x_exp = np.array([x_random**i for i in range(polynomial_length )])
    y = p(x_exp)
    return x_exp, y

def update(polynomial_length,polynomial_count,x_random,y):
    learning_rate = 0.01
    a_bgd = np.random.randint(-100, 300, polynomial_length).reshape(1, polynomial_length)
    def p_new(a_new, x):
        return np.sum(np.matmul(a_new, x), axis=0)

    def error_calc(y_new_val, y):
        return (np.sum((y_new_val - y)**2)) / (polynomial_count)
    
    for i in range(100000):
        y_new_val_bgd = p_new(a_bgd, x_random)
        error_bgd = error_calc(y_new_val_bgd, y)
        gradient_b = (np.sum(2 * (y_new_val_bgd - y) * x_random, axis=1)) / (polynomial_length)
        grad_a1 = gradient_b * learning_rate
        a_bgd = a_bgd - grad_a1

        if error_bgd <=1e-20:
            break
    
    return a_bgd,error_bgd