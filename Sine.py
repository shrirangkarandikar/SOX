import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm 
import math

# Initialize global variables for initial weights, weight updates, and error values
initial_weights = None
weight_updates = []
error_mat = []


def sine(dimensions, num_func,datapoints):

    x_orr = np.random.uniform(-1, 1, datapoints)
    #print(x_orr)
    x_random= np.matrix([(((-1)**(i+1))*(x_orr**((2*i)-1))) for i in range(1,dimensions)]).T
    #x_random=np.matrix([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18],[19,20,21],[22,23,24]])
    #print("x",x_random)
    #print(x_random)
    #print('******************************')
    # Initialize weights randomly
    weights = np.matrix([1/math.factorial((2*i)-1) for i in range(1,dimensions)]).T
    y_target=np.dot(x_random,weights)
    # Compute the target values as a linear combination of inputs and weights, followed by sigmoid activation
    y =np.array(np.sin(x_orr))
    #print("y",y)   
    #print("t",y_target) 
    learning_rate1 = np.matrix([[0.01], [0.001], [0.0001]])
    learning_rate2 = np.matrix([[0.01], [0.0001], [0.00001]])
    learning_rate3 = np.matrix([[0.001], [0.00001], [0.000001]])
    #weights_n = np.random.uniform(0, 1, (dimensions-1, num_func))
    weights_n = (np.matrix([1/math.factorial((2*i)-1) for i in range(1,dimensions)])*(0.7)).T
    #weights_n=np.ones(dimensions-1, num_func)
    #print("wo",weights,weights_n)
    def p1(x,weights_n):
        return np.dot(x, weights_n)
    def calculate_errors(z):
        return np.square(z)/datapoints
    for i in range(100000):
        y_mult = p1(x_random,weights_n)
        #print("y",y_mult)
        z = y_mult - y_target        
        #print("z",z)
        error = calculate_errors(z)
        if i<3000:
            grad_a1 = np.dot(z.T,x_random)*learning_rate1/datapoints
        elif(i>=3000 and i<6000):
            grad_a1 = np.dot(z.T,x_random)*learning_rate2/datapoints 
        else:
            grad_a1 = np.dot(z.T,x_random)*learning_rate3/datapoints 
        #print(f"iteration {i}")
        #print(grad_a1)
        #print("g",grad_a1)          
        weights_n -= grad_a1.T
        #print("w",weights_n)
        
        # Check if the error has fallen below the tolerance
        if np.sum(error) < 0.00001:
            break
    
    
    return error,y_mult,x_orr, y_target,i
    #return error, y_target, x_orr

error, y_new, x, y_orig,iterations = sine(4,1,80)

print(iterations)
#print(x.shape)

plt.figure()
y = [i for i in y_new]
y_o = [i for i in y_orig]
plt.scatter(x.T, y_o)
#print(x.T.shape, y_new.shape, y_new.reshape(x.T.shape).shape)
plt.scatter(x.T, y)
'''
plt.show()
plt.figure()
y = [i for i in y_orig]
plt.scatter(x.T, y)
'''
plt.show()
