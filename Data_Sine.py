import numpy as np
import math


def create(dimensions, num_func, datapoints):
    x_orr = np.random.uniform(-4, 4, datapoints)
    x_random = np.matrix([(((-1)**(i+1))*(x_orr**((2*i)-1))) for i in range(1, dimensions)]).T
    weights = np.matrix([1/math.factorial((2*i)-1) for i in range(1, dimensions)]).T
    y_target = np.dot(x_random, weights)
    #y = np.array(np.sin(x_orr))    
    weights_n = (np.matrix([1/math.factorial((2*i)-1) for i in range(1, dimensions)])*(0.7)).T
    return x_random,y_target,weights_n





