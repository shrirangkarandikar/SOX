import Func_Fit
from Func_Fit import playground

'''

    Func_Fit is a module for function fitting using polynomial and monomial models. It can generate random input 
    data and uses gradient descent to minimize errors and adjust weights for either general polynomials 
    (multiple variables) or monomials (single-variable exponentials). The module allows switching between 
    fitting modes via monomial(1/0), and users can customize data by providing their own inputs.

    Let the function be a linear function made of n dimension vectors.
__________________________________________________________________________________________________________________
    1] .create_data(dimensions,num_func,monomial,datapoints):

    If data is not available, user can generate data using .create_data(). The size of the data can be determined 
    by the user. The data generates the x values required for the polynomial and their coeffients. It calculates
    the value for the polynomial and returns it as y.
-------------------------------------------------------------------------------------
    1) dimension: The number of variables (x) in the polynomial. This determines the number of terms 
    (e.g., x1, x2, ..., xn) needed to compute the polynomial.

    2) datapoints: The number of distinct sets of data points (P1, P2, P3, ...). 
    This represents how many different polynomials need to be evaluated.

    3) num_functions: The number of functions (y) to be computed for a single set of data points. 
    This is essentially the number of output values (y1, y2, ..., yn).

    4)monomial: Determines whether the polynomials should be single-variable exponentials or more complex. 
    If set to 1, the polynomials are monomials (single-variable exponentials, e.g., x^n). 
    If set to 0, the polynomials can include mixed or multi-variable terms.
_________________________________________________________________________________________________________________
    2] .fit(x_random,y,error_tol)

    This function genrates a random polynomial and updates the coeffiecients and maps it to the polynomial we 
    generated using gradient descent algorithm. If the user already possesses the data of the input and the
    required output user can skip .create_data(). 
-------------------------------------------------------------------------------------
    1) x_random: It is the data or the values of the variables in polynomial.(x1,x2,x3...)

    2) y: The value of the polynomial computed with x_random(y1,y2,y3...)

    3) error_tol: The error which can be tolerated by the new function.
__________________________________________________________________________________________________________________
    3] .animate()

    If the user wants to visualize the process under .fit(), they can use .animate(). It provides animation of the
    coeffiecients of the polynomials and of error convergence towards zero.
        
_______________________________________________________________________________________________________________________
    Following is the demo code for the module Func_fit:
'''
'''
print("Enter data info: ")
dimensions=int(input("Dimension of the polynomial "))
num_func=int(input("Number of Functions "))
datapoints=int(input("Count of the Function value "))                                                                                                                                                                                                                                                
monomial=int(input("Monomial:(1/0)"))
error_tol=float(input("Tolerable error "))

x_random,y=Func_Fit.create_data(dimensions,num_func,monomial,datapoints)
print(x_random)
error = Func_Fit.fit(x_random,y,error_tol)
anim = Func_Fit.animate()
'''

'''


#The following is the tutorial of the working of the Func_fit

t=Func_Fit.tutorial()
t.run()
'''

'''
The Playground class provides a suite of functions for experimenting with gradient descent and other optimization methods. 
It includes functionalities for testing different learning rates, comparing gradient descent and stochastic gradient descent, 
and evaluating how varying input sizes affect convergence. Additionally, it provides error metrics to assess the performance 
of optimization algorithms.
--------------------------------------------------------------------------------------------------------------------
Functions:
1)learning_rate(dimensions, num_func, datapoints):
    Analyzes the effect of different learning rates on the convergence of gradient descent. Compares low and high learning 
    rates by plotting their error convergence over iterations.
    
    dimensions: Number of variables in the polynomial.
    num_func: Number of output functions.
    datapoints: Number of data points to use for testing.
------------------------------------------------------------------------------------------------------------------------
2)Grad_Desc(dimensions, num_func, datapoints):
    Compares Gradient Descent (GD) and Stochastic Gradient Descent (SGD) methods. Visualizes how each method converges by 
    plotting error convergence over iterations.
    
    dimensions: Number of variables in the polynomial.
    num_func: Number of output functions.
    datapoints: Number of data points to use for testing.
----------------------------------------------------------------------------------------------------------------------------
3)var_inp(dimensions, num_func, datapoints):
    Examines the effect of varying input sizes on gradient descent convergence. Compares the convergence of GD for different 
    input sizes by plotting error convergence.
    
    dimensions: Number of variables in the polynomial.
    num_func: Number of output functions.
    datapoints: Number of data points to use for testing.
----------------------------------------------------------------------------------------------------------------------------------------
4)error(dimensions, num_func, datapoints):
    Tests the performance of different error metrics (MAE, MSE, and infinity norm) in gradient descent. Analyzes how each metric
    influences the convergence of gradient descent by visualizing error convergence over iterations.
    
    dimensions: Number of variables in the polynomial.
    num_func: Number of output functions.
    datapoints: Number of data points to use for testing.
--------------------------------------------------------------------------------------------------------------------------------------
5)tolerance(dimensions, num_func, datapoints):
    Tests the performance when the tolerance for error is changed. Analyses how it affects the weights of the 
    functions.

    dimensions: Number of variables in the polynomial.
    num_func: Number of output functions.
    datapoints: Number of data points to use for testing.
'''

learning_rate_fun=Func_Fit.playground.learning_rate(dimensions=5, num_func=1, datapoints=100)
#errorchange=Func_Fit.playground.error(dimensions=5, num_func=1, datapoints=10)
#grad=Func_Fit.playground.Grad_Desc(dimensions=5, num_func=1, datapoints=10,batch_size=5)
#var_inp=Func_Fit.playground.var_inp(dimensions=5, num_func=1, datapoints=10)
#tol=Func_Fit.playground.tolerance(dimensions=5, num_func=1, datapoints=10)
#activ=Func_Fit.playground.activation(dimensions=5,num_func=1,datapoints=10)
