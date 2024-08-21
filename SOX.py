import Func_Fit

'''

    Func_Fit is a module for function fitting using polynomial and monomial models. It can generate random input 
    data and uses gradient descent to minimize errors and adjust weights for either general polynomials 
    (multiple variables) or monomials (single-variable exponentials). The module allows switching between 
    fitting modes via a selector (sel), and users can customize data by providing their own inputs.

    Let the function be a linear function P(x)=y1(x)+y2(x)+y3(x)+y4(x)...=a1.x1+a2.x2+a3.x3+a4.x4...an.xn
__________________________________________________________________________________________________________________
    1] .create_data(dimensions,val_count,num_func,monomial):

    If data is not available, user can generate data using .create_data(). The size of the data can be determined 
    by the user. The data generates the x values required for the polynomial and their coeffients. It calculates
    the value for the polynomial and returns it as y.
-------------------------------------------------------------------------------------
    1) dimension: The number of variables (x) in the polynomial. This determines the number of terms 
    (e.g., x1, x2, ..., xn) needed to compute the polynomial.

    2) val_count: The number of distinct sets of data points (P1, P2, P3, ...). 
    This represents how many different polynomials need to be evaluated.

    3) num_functions: The number of functions (y) to be computed for a single set of data points. 
    This is essentially the number of output values (y1, y2, ..., yn).

    4)monomial: Determines whether the polynomials should be single-variable exponentials or more complex. 
    If set to 1, the polynomials are monomials (single-variable exponentials, e.g., x^n). 
    If set to 0, the polynomials can include mixed or multi-variable terms.
_________________________________________________________________________________________________________________
    2] .fit(dimensions,num_functions,x_random,y)

    This function genrates a random polynomial and updates the coeffiecients and maps it to the polynomial we 
    generated using gradient descent algorithm. If the user already possesses the data of the input and the
    required output user can skip .create_data(). 
-------------------------------------------------------------------------------------
    1) x_random: It is the data or the values of the variables in polynomial.(x1,x2,x3...)

    2) y: The value of the polynomial computed with x_random(y1,y2,y3...)
__________________________________________________________________________________________________________________
    3] .animate()

    If the user wants to visualize the process under .fit(), he can use .animate(). It provides animation of the
    coeffiecients of the polynomials and of error convergence towards zero.
        
_______________________________________________________________________________________________________________________
    
'''

print("Enter data info: ")
dimensions=int(input("Dimension of the polynomial "))
num_func=int(input("Number of Functions "))
val_count=int(input("Count of the Function value "))                                                                                                                                                                                                                                                
monomial=int(input("Monomial:(1/0)"))

x_random,y=Func_Fit.create_data(dimensions,val_count,num_func,monomial)
error = Func_Fit.fit(dimensions, num_func,x_random,y)
anim = Func_Fit.animate()

print("Error for the Polynomial",error)

