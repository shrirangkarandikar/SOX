import numpy as np
import matplotlib.pyplot as plt

def create_data(mean, variance, dimensions, datapoints):
    dev = (variance)**0.5
    x_random = np.random.normal(mean, dev, (datapoints, dimensions))
    weights = np.random.uniform(0, 2, (dimensions, 1))
    y_target = np.dot(x_random, weights)
    return y_target, x_random

def fit(x_random, y_target):
    dimensions = len(x_random[0])
    learning_rate = 0.00001
    weights_n = np.random.uniform(0, 2, (dimensions, 1))
    errors = []  # Track errors for plotting
    
    def p1(x):
        return np.dot(x, weights_n)
    
    def calculate_errors(z):
        return ((z)**2) / dimensions
    
    for i in range(100000):
        y_mult = p1(x_random)
        z = y_mult - y_target
        error = calculate_errors(z)
        errors.append(np.sum(error))  # Track the total error at each iteration
        grad_a1 = np.dot(x_random.T, z * y_mult) * learning_rate / dimensions
        weights_n -= grad_a1
        if np.all(error < 0.001):
            break
    
    print(f"Converged in {i} iterations")
    return errors  # Return errors across iterations for plotting

# Generate data and fit models, collecting errors
y1, x1 = create_data(1, 1, 5, 100)
errors1 = fit(x1, y1)

y2, x2 = create_data(1, 0, 5, 100)
errors2 = fit(x2, y2)

y3, x3 = create_data(0, 0.4, 5, 100)
errors3 = fit(x3, y3)

y4, x4 = create_data(0, 0.1, 5, 100)
errors4 = fit(x4, y4)

# Plotting the errors for each scenario in separate subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Error Reduction Over Iterations for Different Data Scenarios')

# Subplot 1: Mean=1, Var=1
axs[0, 0].plot(errors1, label="Mean=1, Var=1", color="b")
axs[0, 0].set_title("Mean=1, Var=1")
axs[0, 0].set_xlabel("Iterations")
axs[0, 0].set_ylabel("Error")
axs[0, 0].legend()

# Subplot 2: Mean=1, Var=0
axs[0, 1].plot(errors2, label="Mean=1, Var=0", color="g")
axs[0, 1].set_title("Mean=1, Var=0")
axs[0, 1].set_xlabel("Iterations")
axs[0, 1].set_ylabel("Error")
axs[0, 1].legend()

# Subplot 3: Mean=0, Var=0.4
axs[1, 0].plot(errors3, label="Mean=0, Var=0.4", color="r")
axs[1, 0].set_title("Mean=0, Var=0.4")
axs[1, 0].set_xlabel("Iterations")
axs[1, 0].set_ylabel("Error")
axs[1, 0].legend()

# Subplot 4: Mean=0, Var=0.1
axs[1, 1].plot(errors4, label="Mean=0, Var=0.1", color="m")
axs[1, 1].set_title("Mean=0, Var=0.1")
axs[1, 1].set_xlabel("Iterations")
axs[1, 1].set_ylabel("Error")
axs[1, 1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title
plt.show()
