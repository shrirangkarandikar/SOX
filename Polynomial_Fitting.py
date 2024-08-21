import numpy as np


def fit(polynomial_length,polynomial_count,polynomial_outputs,sel):
    if(sel==0):
        def polynomial(polynomial_length,polynomial_count,polynomial_outputs):
            def create_data(polynomial_length,polynomial_count,polynomial_outputs):
                
                '''
                polynomial_length:dimensions, polynomial_count:num_functions,polynomial_outputs: value_count
                Polynomial
                x_random : Input of the polynomials, shape(x_count * data_points)
                weights : Weights of the final polynomial, shape(x_count * y_Counts)
                p(x) : Computes the final polynomial
                y : Final Polynomial, shape(y_count*data_points)
                '''
                x_random = np.random.uniform(-1, 1, (polynomial_length, polynomial_count))  
                x_random = x_random.T
                weights = np.random.uniform(0, 10, (polynomial_length, polynomial_outputs))  
                
                def p(x):
                    return np.dot(x, weights)
                y = p(x_random)
                return x_random,y
            x_random,y=create_data(polynomial_length,polynomial_count,polynomial_outputs)
            def update(polynomial_length, polynomial_outputs,x_random,y):
                
                '''
                y_mult : Fitting Polynomial, shape(y_count*data_points)
                weights_n : Initial weights for the curve to be adjusted to fit Final Polynomial
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
                return error
            error = update(polynomial_length, polynomial_outputs,x_random,y)
            return error
        error=polynomial(polynomial_length,polynomial_count,polynomial_outputs)
        error=np.sum(error)
        return error
    elif(sel):
        def monomial(polynomial_length,polynomial_count,polynomial_outputs):
            
            '''
            For a monomial of degree n.

            polynomial=n 

            '''
            def data():    
                
                weights = np.random.uniform(0, 10, (polynomial_length, polynomial_outputs)) 
                def p(x):
                    return np.dot(x, weights)
                x_random = np.random.uniform(-1, 1, polynomial_count)
                x_exp = np.array([x_random**i for i in range(polynomial_length )])
                y = p(x_exp)
                return x_exp, y
            x_random, y=data()
            def conv():
                learning_rate = 0.01
                a_bgd = np.random.uniform(0, 10, (polynomial_length, polynomial_outputs))
                def p_new(weights_n, x):
                    return np.dot(x, weights_n)

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
                
                return error_bgd
            error=conv()
            return error
        error = monomial(polynomial_length,polynomial_count,polynomial_outputs)
        return error 