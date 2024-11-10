import numpy as np

def create_data(dimensions, num_func, monomial_sel, datapoints):
    if monomial_sel == 0:
        # Generate random input data for multiple variables (polynomials)
        x_random = np.random.uniform(-1, 1, (datapoints, dimensions))
    elif monomial_sel == 1:
        # Generate random input data for a single variable raised to different powers (monomials)
        x_random = np.random.uniform(-1, 1, datapoints)
        x_random = np.array([x_random**i for i in range(dimensions)]).T

    # Initialize weights randomly
    weights = np.random.uniform(0, 2, (dimensions, num_func))   

    # Compute the target values as a linear combination of inputs and weights, followed by sigmoid activation
    y = np.dot(x_random, weights)
    y = 1 / (1 + np.exp(-y))
    return x_random, y

x_random, y=create_data(10,10,1,100)  

def fit(x_random, y, error_tol):

    dimensions = len(x_random[0])
    num_func = len(y[0])

    learning_rate = 0.01
    weights_n = np.random.uniform(0, 2, (dimensions, num_func))

    # Define the polynomial function as a dot product of inputs and weights
    def p1(x):
        return np.dot(x, weights_n)
    
    # Function to calculate the error as mean squared error
    def calculate_errors(z):
        return ((z)**2) / dimensions
    
    def relu_activation(z):
            if(z<=0):
                 val=0
            elif(z>0 and z<1):
                 val=z**0.5
            else:
                 val=z**1.1
            return val

    def relu_derivative(z):
            return (z > 0).astype(float)

    # Gradient descent loop
    for i in range(100000):
        # Compute predictions using sigmoid activation
        y_mult = relu_activation(p1(x_random))
        
        # Calculate the error and the gradient
        z = y_mult - y 
        error = calculate_errors(z)
        grad_a1 = np.dot(x_random.T, z * relu_derivative(y_mult)) * learning_rate / dimensions
        #update the weights        
        weights_n -= grad_a1
        
        # Check if the error has fallen below the tolerance
        if np.all(error < error_tol):
            break
        
    
    print(f"Converged in {i} iterations")
    error_all = np.sum(error)
    return error_all

error=fit(x_random,y,0.01)