import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize global variables for initial weights, weight updates, and error values
initial_weights = None
weight_updates = []
error_mat = []

# Function to generate input data and corresponding polynomial outputs
def create_data(polynomial_length, polynomial_count, polynomial_outputs, sel):
    if sel == 0:
        # Generate random input data for a general polynomial (multiple variables)
        x_random = np.random.uniform(-1, 1, (polynomial_length, polynomial_count))  
        x_random = x_random.T
        # Generate random initial weights
        weights = np.random.uniform(0, 10, (polynomial_length, polynomial_outputs))
        fit.initial_weights = weights.copy()
        
        # Define the polynomial function
        def p(x):
            return np.dot(x, weights)
        
        # Compute polynomial outputs (y) based on input data
        y = p(x_random)
        return x_random, y
    
    elif sel == 1:
        # Generate random input data for a monomial (single-variable exponentials)
        weights = np.random.uniform(0, 10, (polynomial_length, polynomial_outputs)) 
        fit.initial_weights = weights.copy()
        
        # Define the polynomial function
        def p(x):
            return np.dot(x, weights)
        
        # Generate single-variable exponentials for input data
        x_random = np.random.uniform(-1, 1, polynomial_count)
        x_exp = np.array([x_random**i for i in range(polynomial_length)])
        
        # Compute polynomial outputs (y) based on input data
        y = p(x_exp)
        return x_exp, y

# Function to fit the polynomial by adjusting the weights using gradient descent
def fit(polynomial_length, polynomial_outputs, x_random, y):
    learning_rate = 0.01  # Learning rate for gradient descent
    weights_n = np.random.uniform(0, 10, (polynomial_length, polynomial_outputs))  # Initialize random weights
    fit.weight_updates = []  # List to store weight updates at each iteration
    fit.error_mat = []  # List to store error values at each iteration
    
    # Define the polynomial function
    def p1(x):
        return np.dot(x, weights_n)
    
    # Function to calculate the error
    def calculate_errors(z):
        return ((z)**2) / polynomial_length
    
    # Gradient descent loop
    for i in range(100000):
        y_mult = p1(x_random)  # Compute the polynomial outputs with current weights
        z = y_mult - y  # Compute the difference between predicted and actual outputs
        error = calculate_errors(z)  # Calculate the error
        grad_a1 = np.dot(x_random.T, z) * learning_rate / polynomial_length  # Compute the gradient
        weights_n -= grad_a1  # Update the weights using the gradient
        
        # Check if the error is below the tolerance level
        if np.all(error < 1e-2):
            break
        
        fit.weight_updates.append(weights_n.copy())  # Store the updated weights
        fit.error_mat.append(np.sum(error))  # Store the sum of errors
    
    print(f"Converged in {i} iterations")
    error_all = np.sum(error)
    return error_all

# Function to animate the weight updates and error convergence during the fitting process
def animate():
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Create a figure with two subplots
    
    # Set up the left plot for weight updates
    ax[0].set_title("Weight Updates")
    ax[0].set_xlabel("Weight Index")
    ax[0].set_ylabel("Weight Value")
    ax[0].set_xlim(0, 1000)
    ax[0].set_ylim(0, 10)
    
    # Set up the right plot for error convergence
    ax[1].set_xlim(0, len(fit.weight_updates))
    ax[1].set_ylim(0, 1.1 * max(fit.error_mat))
    line_error, = ax[1].plot([], [], lw=2, color='green', label="Error")
    ax[1].set_title("Error Convergence")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Error Value")
    ax[1].legend()
    
    # Function to update the animation at each frame
    def update_animation(i):
        # Update the left plot with current weight updates
        ax[0].clear()
        ax[0].set_title(f"Weight Updates (Iteration {i})")
        ax[0].set_xlabel("Weight Index")
        ax[0].set_ylabel("Weight Value")
        initial_weights = fit.initial_weights.flatten()
        current_weights = fit.weight_updates[i].flatten()
        ax[0].scatter(range(len(initial_weights)), initial_weights, color='blue', label="Initial Weights")
        ax[0].scatter(range(len(current_weights)), current_weights, color='red', label="Current Weights")
        ax[0].legend()

        # Update the right plot with current error convergence
        ax[1].clear()
        ax[1].plot(range(i + 1), fit.error_mat[:i + 1], color='green', lw=2)
        ax[1].set_xlim(0, len(fit.weight_updates))
        ax[1].set_ylim(0, 1.1 * max(fit.error_mat))
        ax[1].set_title("Error Convergence")
        ax[1].set_xlabel("Iteration")
        ax[1].set_ylabel("Error Value")
        ax[1].legend()
    
    # Create the animation using the update function
    anim = FuncAnimation(fig, update_animation, frames=len(fit.weight_updates), interval=50, repeat=False)
    plt.show()
