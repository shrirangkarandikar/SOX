import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


initial_weights = None
weight_updates = []
error_mat = []

def create_data(polynomial_length, polynomial_count, polynomial_outputs, sel):
    if sel == 0:
        x_random = np.random.uniform(-1, 1, (polynomial_length, polynomial_count))  
        x_random = x_random.T
        weights = np.random.uniform(0, 10, (polynomial_length, polynomial_outputs))
        fit.initial_weights = weights.copy()
        def p(x):
            return np.dot(x, weights)
        y = p(x_random)
        return x_random, y
    
    elif sel == 1:
        weights = np.random.uniform(0, 10, (polynomial_length, polynomial_outputs)) 
        fit.initial_weights = weights.copy()
        def p(x):
            return np.dot(x, weights)
        x_random = np.random.uniform(-1, 1, polynomial_count)
        x_exp = np.array([x_random**i for i in range(polynomial_length)])
        y = p(x_exp)
        return x_exp, y

def fit(polynomial_length, polynomial_outputs, x_random, y):
    learning_rate = 0.01
    weights_n = np.random.uniform(0, 10, (polynomial_length, polynomial_outputs))
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
        if np.all(error < 1e-2):
            break
        fit.weight_updates.append(weights_n.copy())
        fit.error_mat.append(np.sum(error))
    
    print(f"Converged in {i} iterations")
    error_all = np.sum(error)
    return error_all

def animate():
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left plot for weight updates
    ax[0].set_title("Weight Updates")
    ax[0].set_xlabel("Weight Index")
    ax[0].set_ylabel("Weight Value")
    ax[0].set_xlim(0, 1000)
    ax[0].set_ylim(0, 10)
    # Right plot for error convergence
    ax[1].set_xlim(0, len(fit.weight_updates))
    ax[1].set_ylim(0, 1.1 * max(fit.error_mat))
    line_error, = ax[1].plot([], [], lw=2, color='green', label="Error")
    ax[1].set_title("Error Convergence")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Error Value")
    ax[1].legend()
    
    def update_animation(i):
        # Update left plot with weight updates
        ax[0].clear()
        ax[0].set_title(f"Weight Updates (Iteration {i})")
        ax[0].set_xlabel("Weight Index")
        ax[0].set_ylabel("Weight Value")
        initial_weights = fit.initial_weights.flatten()
        current_weights = fit.weight_updates[i].flatten()
        ax[0].scatter(range(len(initial_weights)), initial_weights, color='blue', label="Initial Weights")
        ax[0].scatter(range(len(current_weights)), current_weights, color='red', label="Current Weights")
        ax[0].legend()

        # Update right plot with error convergence
        ax[1].clear()
        ax[1].plot(range(i + 1), fit.error_mat[:i + 1], color='green', lw=2)
        ax[1].set_xlim(0, len(fit.weight_updates))
        ax[1].set_ylim(0, 1.1 * max(fit.error_mat))
        ax[1].set_title("Error Convergence")
        ax[1].set_xlabel("Iteration")
        ax[1].set_ylabel("Error Value")
        ax[1].legend()
    
    anim = FuncAnimation(fig, update_animation, frames=len(fit.weight_updates), interval=50, repeat=False)
    plt.show()