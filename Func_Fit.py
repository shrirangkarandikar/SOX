import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


initial_weights = None
weight_updates = []
error_mat = []

def create_data(polynomial_length, polynomial_outputs, sel, polynomial_count,):
    if sel == 0:
        x_random = np.random.uniform(-1, 1, (polynomial_count,polynomial_length))
    elif sel == 1:
        x_random = np.random.uniform(-1, 1, polynomial_count)
        x_random = np.array([x_random**i for i in range(polynomial_length)]).T
    weights = np.random.uniform(0, 2, (polynomial_length, polynomial_outputs))
    fit.initial_weights = weights
    y = np.dot(x_random, weights)
    return x_random, y
    

def fit(x_random, y,error_tol):
    polynomial_length=len(x_random[0])
    polynomial_outputs=len(y[0])

    learning_rate = 0.01
    weights_n = np.random.uniform(0, 2, (polynomial_length, polynomial_outputs))
    fit.weight_updates = []
    fit.error_mat = []
    
    def p1(x):
        return np.dot(x, weights_n)
    
    def calculate_errors(z):
        return ((z)**2) / polynomial_length
    
    for i in range(100000):
        y_mult = p1(x_random)
        z = y_mult - y 
        error = calculate_errors(z)
        grad_a1 = np.dot(x_random.T, z) * learning_rate / polynomial_length        
        weights_n -= grad_a1
        if np.all(error < error_tol):
            break
        fit.weight_updates.append(weights_n.copy())
        fit.error_mat.append(np.sum(error))
    
    print(f"Converged in {i} iterations")
    print("initial weights")
    print(fit.initial_weights)
    print("final weights")
    print(weights_n)
    print("difference")
    print(fit.initial_weights - weights_n)
    print(np.max(fit.initial_weights - weights_n), 
          np.argmax(fit.initial_weights - weights_n))
    print("error")
    print(error)
    error_all = np.sum(error)
    return error_all



def animate():
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Left plot for weight updates
    ax[0].set_title("Weight Updates")
    ax[0].set_xlabel("Weight Index")
    ax[0].set_ylabel("Weight Value")
    ax[0].set_xlim(0, len(fit.initial_weights.flatten()))
    ax[0].set_ylim(0, 5)
    arrows=ax[0].quiver([],[],[]) 
  

    initial_weights = fit.initial_weights.flatten()
    current_weights, = ax[0].plot([], [], 'ro')
    ax[0].scatter(range(len(initial_weights)), initial_weights, color='blue')

    # Right plot for error convergence
    ax[1].set_xlim(0, len(fit.error_mat))
    ax[1].set_ylim(min(fit.error_mat) * 0.1, max(fit.error_mat) * 10)
    ax[1].set_yscale('log')
    ax[1].set_title("Error Convergence")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Error Value")

    line_error, = ax[1].plot([], [], lw=2, color='green')

    def update_animation(i):
        
        # Update current weights in the left plot
        current_weights.set_data(range(len(initial_weights)), fit.weight_updates[i].flatten())        
        #arrows.set_UVC([len(weight_updates),weight_updates],len(initial_weights),initial_weights)
        # Update error convergence in the right plot
        line_error.set_data(range(i + 1), fit.error_mat[:i + 1])

        return current_weights, line_error,arrows,

    # Set blit=True for faster animation
    anim = FuncAnimation(fig, update_animation, frames=len(fit.weight_updates), interval=1, blit=True, repeat=False)
    plt.show()

    

class playground():
    def learning_rate(polynomial_length, polynomial_outputs, polynomial_count,err):
        # Generate random input data and initial weights
        x_random = np.random.uniform(-1, 1, (polynomial_count, polynomial_length))
        weights = np.random.uniform(0, 2, (polynomial_length, polynomial_outputs))
        y = np.dot(x_random, weights)
        
        # Initialize weights for each strategy
        np.random.seed(4)
        weights_n = np.random.uniform(0, 2, (polynomial_length, polynomial_outputs))
        
        # Function for low learning rate
        def low():
            lr = 0.00001
            weights_n_low = weights_n.copy()
            error_list_low = []
            
            def p1(x):
                return np.dot(x, weights_n_low)
            
            def calculate_errors(z):
                return np.mean(z**2)  # Use mean squared error
            
            for i in range(100000):
                y_mult = p1(x_random)
                z = y_mult - y
                error = calculate_errors(z)
                error_list_low.append(error)
                grad_a1 = np.dot(x_random.T, z) * lr / 10
                weights_n_low -= grad_a1
                
                if error < err:
                    break
            
            return y_mult, error_list_low
        
        y_low, error_list_low = low()
        
        # Function for high learning rate
        def high():
            lr = 0.1
            weights_n_high = weights_n.copy()
            error_list_high = []
            
            def p1(x):
                return np.dot(x, weights_n_high)
            
            def calculate_errors(z):
                return np.mean(z**2)  # Use mean squared error
            
            for i in range(100000):
                y_mult = p1(x_random)
                z = y_mult - y
                error = calculate_errors(z)
                error_list_high.append(error)
                grad_a1 = np.dot(x_random.T, z) * lr / 10
                weights_n_high -= grad_a1
                
                if error < err:
                    break
            
            return y_mult, error_list_high
        
        y_high, error_list_high = high()
        
        # Function for adaptive learning rate using Adagrad
        def adap():
            lr = 0.00001
            epsilon = 1e-8
            weights_n_adap = weights_n.copy()
            error_list_adap = []
            grad_accumulator = np.zeros_like(weights_n_adap)
            
            def p1(x):
                return np.dot(x, weights_n_adap)
            
            def calculate_errors(z):
                return np.mean(z**2)  # Use mean squared error
            
            for i in range(100000):
                y_mult = p1(x_random)
                z = y_mult - y
                error = calculate_errors(z)
                error_list_adap.append(error)
                grad_a1 = np.dot(x_random.T, z) / 10
                grad_accumulator += grad_a1 ** 2
                adjusted_lr = lr / (np.sqrt(grad_accumulator) + epsilon)
                weights_n_adap -= adjusted_lr * grad_a1
                
                if error < err:
                    break
            
            return y_mult, error_list_adap
        
        y_adap, error_list_adap = adap()
        
        # Plot the error convergence for all three strategies
        plt.figure(figsize=(12, 6))
        plt.plot(error_list_low, label='Low Learning Rate', color='green')
        plt.plot(error_list_high, label='High Learning Rate', color='blue')
        plt.plot(error_list_adap, label='Adaptive Learning Rate', color='red')
        plt.yscale('log')  # Use a logarithmic scale to better visualize the error
        plt.xlabel('Iteration')
        plt.ylabel('Error (MSE)')
        plt.title('Error Convergence for Different Learning Rates')
        plt.legend()
        plt.show()
        
        # Plot the final output of the models with different learning rates
        plt.figure(figsize=(12, 6))
        plt.plot(x_random, y, 'o', label='True Values', color='black')
        plt.plot(x_random, y_low, label='Low Learning Rate', color='green')
        plt.plot(x_random, y_high, label='High Learning Rate', color='blue')
        plt.plot(x_random, y_adap, label='Adaptive Learning Rate', color='red')
        plt.xlabel('Input Features')
        plt.ylabel('Output Values')
        plt.title('Model Outputs with Different Learning Rates')
        plt.legend()
        plt.show()
        
    
        
        def error(polynomial_length, polynomial_outputs,polynomial_count):
            x_random = np.random.uniform(-1, 1, (polynomial_count,polynomial_length))
            weights = np.random.uniform(0, 2, (polynomial_length, polynomial_outputs))
            y = np.dot(x_random, weights)
            np.seed(4)
            weights_n = np.random.uniform(0, 2, (polynomial_length, polynomial_outputs))
            def MAE(weights_n):
                y = np.dot(x_random, weights)
                lr = 0.1
                def p1(x):
                    return np.dot(x, weights_n)                
                def calculate_errors(z):
                    return z / polynomial_length               
                for i in range(100000):
                    y_mult = p1(x_random)
                    z = y_mult - y 
                    error = calculate_errors(z)
                    grad_a1 = np.dot(x_random.T, z) * lr /10       
                    weights_n -= grad_a1
                    if np.all(error < 0.01):
                        break
                return y_mult
            y_MAE=MAE(weights_n)
            np.seed(4)
            weights_n = np.random.uniform(0, 2, (polynomial_length, polynomial_outputs))
            def MSE(weights_n):
                y = np.dot(x_random, weights)
                lr = 0.1
                def p1(x):
                    return np.dot(x, weights_n)                
                def calculate_errors(z):
                    return z**2 / polynomial_length               
                for i in range(100000):
                    y_mult = p1(x_random)
                    z = y_mult - y 
                    error = calculate_errors(z)
                    grad_a1 = np.dot(x_random.T, z) * lr /10       
                    weights_n -= grad_a1
                    if np.all(error < 0.01):
                        break
                return y_mult
            y_MSE=MSE(weights_n)
            def error_norm(weights_n):
                y = np.dot(x_random, weights)
                lr = 0.1
                def p1(x):
                    return np.dot(x, weights_n)                
                def calculate_errors(z):
                    return z**2 / polynomial_length               
                for i in range(100000):
                    y_mult = p1(x_random)
                    z = y_mult - y 
                    error = calculate_errors(z)
                    grad_a1 = np.dot(x_random.T, z) * lr /10       
                    weights_n -= grad_a1
                    if np.all(error < 0.01):
                        break
                return y_mult
            y_MSE=MSE(weights_n)

        def Grad_Desc(polynomial_length, polynomial_outputs,polynomial_count,error):
            x_random = np.random.uniform(-1, 1, (polynomial_count,polynomial_length))
            weights = np.random.uniform(0, 2, (polynomial_length, polynomial_outputs))
            y = np.dot(x_random, weights)
            np.seed(4)
            weights_n = np.random.uniform(0, 2, (polynomial_length, polynomial_outputs))
            def GD(weights_n):
                y = np.dot(x_random, weights)
                lr = 0.1
                def p1(x):
                    return np.dot(x, weights_n)                
                def calculate_errors(z):
                    return z / polynomial_length               
                for i in range(100000):
                    y_mult = p1(x_random)
                    z = y_mult - y 
                    error = calculate_errors(z)
                    grad_a1 = np.dot(x_random.T, z) * lr /10       
                    weights_n -= grad_a1
                    if np.all(error < 0.01):
                        break
                return y_mult
            
            np.seed(4)
            weights_n = np.random.uniform(0, 2, (polynomial_length, polynomial_outputs))
            def SGD(weights_n):
                learning_rate = 0.1
                batch_size = 10
                num_iterations = 100000
                error_tol = 0.01

                def p1(x):
                    return np.dot(x, weights_n)
                
                def calculate_errors(z):
                    return np.mean(z**2)  # Use mean squared error for batch updates
                
                num_samples = x_random.shape[0]

                for i in range(num_iterations):
                    # Shuffle the data for each iteration
                    indices = np.random.permutation(num_samples)
                    x_random_shuffled = x_random[indices]
                    y_shuffled = y[indices]
                    
                    for start_idx in range(0, num_samples, batch_size):
                        end_idx = min(start_idx + batch_size, num_samples)
                        x_batch = x_random_shuffled[start_idx:end_idx]
                        y_batch = y_shuffled[start_idx:end_idx]
                        
                        # Compute predictions and errors for the batch
                        y_mult = p1(x_batch)
                        z = y_mult - y_batch
                        error = calculate_errors(z)
                        
                        # Compute gradients and update weights
                        grad_a1 = np.dot(x_batch.T, z) * learning_rate / batch_size
                        weights_n -= grad_a1
                        
                        if np.mean(error) < error_tol:
                            print(f"Converged in {i} iterations")
                            return p1(x_random)
                        
        def var_inp(polynomial_length, polynomial_outputs,polynomial_count,error):
            
            def original():
                x_random = np.random.uniform(-1, 1, (polynomial_count,polynomial_length))
                weights = np.random.uniform(0, 2, (polynomial_length, polynomial_outputs))
                y = np.dot(x_random, weights)
                np.seed(4)
                weights_n = np.random.uniform(0, 2, (polynomial_length, polynomial_outputs))
                y = np.dot(x_random, weights)
                lr = 0.1
                def p1(x):
                    return np.dot(x, weights_n)                
                def calculate_errors(z):
                    return z / polynomial_length               
                for i in range(100000):
                    y_mult = p1(x_random)
                    z = y_mult - y 
                    error = calculate_errors(z)
                    grad_a1 = np.dot(x_random.T, z) * lr /10       
                    weights_n -= grad_a1
                    if np.all(error < 0.01):
                        break
                return y_mult
            
            np.seed(4)
            weights_n = np.random.uniform(0, 2, (polynomial_length, polynomial_outputs))
        
            def ten_time():
                polynomial_count=polynomial_count*10
                x_random = np.random.uniform(-1, 1, (polynomial_count,polynomial_length))
                weights = np.random.uniform(0, 2, (polynomial_length, polynomial_outputs))
                y = np.dot(x_random, weights)
                np.seed(4)
                weights_n = np.random.uniform(0, 2, (polynomial_length, polynomial_outputs))
                y = np.dot(x_random, weights)
                lr = 0.1
                def p1(x):
                    return np.dot(x, weights_n)                
                def calculate_errors(z):
                    return z / polynomial_length               
                for i in range(100000):
                    y_mult = p1(x_random)
                    z = y_mult - y 
                    error = calculate_errors(z)
                    grad_a1 = np.dot(x_random.T, z) * lr /10       
                    weights_n -= grad_a1
                    if np.all(error < 0.01):
                        break
                return y_mult
            
            def half():
                if(polynomial_count%2==0):
                    polynomial_count=polynomial_count*1
                else:
                    polynomial_count=(polynomial_count+1)/2
                x_random = np.random.uniform(-1, 1, (polynomial_count,polynomial_length))
                weights = np.random.uniform(0, 2, (polynomial_length, polynomial_outputs))
                y = np.dot(x_random, weights)
                np.seed(4)
                weights_n = np.random.uniform(0, 2, (polynomial_length, polynomial_outputs))
                y = np.dot(x_random, weights)
                lr = 0.1
                def p1(x):
                    return np.dot(x, weights_n)                
                def calculate_errors(z):
                    return z / polynomial_length               
                for i in range(100000):
                    y_mult = p1(x_random)
                    z = y_mult - y 
                    error = calculate_errors(z)
                    grad_a1 = np.dot(x_random.T, z) * lr /10       
                    weights_n -= grad_a1
                    if np.all(error < 0.01):
                        break
                return y_mult

            




    
