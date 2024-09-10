import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm 

# Initialize global variables for initial weights, weight updates, and error values
initial_weights = None
weight_updates = []
error_mat = []

'''
The original Func_fit module. This module provides tools for generating random input data, applying 
gradient descent to minimize errors, and adjusting weights to fit polynomial or monomial models. 
It includes functions for creating data, fitting a model to the data using gradient descent, and 
visualizing the results through animation.
'''

def create_data(dimensions, num_func, monomial_sel, datapoints):
    """
    Generates input data and initial weights, and computes the target output values.

    Parameters:
    - dimensions (int): Number of features or variables in the input data.
    - num_func (int): Number of functions (or outputs) to generate.
    - monomial_sel (int): Selection flag to determine the type of data generation:
        - 0: Generate data for polynomials with multiple variables.
        - 1: Generate data for monomials with a single variable raised to different powers.
    - datapoints (int): Number of data points to generate.

    Returns:
    - x_random (numpy.ndarray): Generated input data, where each row represents a data point and each 
      column represents a feature or variable.
    - y (numpy.ndarray): Target output values, processed through a sigmoid function to yield values between 0 and 1.
    """
    if monomial_sel == 0:
        # Generate random input data for multiple variables (polynomials)
        x_random = np.random.uniform(-1, 1, (datapoints, dimensions))
    elif monomial_sel == 1:
        # Generate random input data for a single variable raised to different powers (monomials)
        x_random = np.random.uniform(-1, 1, datapoints)
        x_random = np.array([x_random**i for i in range(dimensions)]).T

    # Initialize weights randomly
    weights = np.random.uniform(0, 2, (dimensions, num_func))
    fit.initial_weights = weights

    # Compute the target values as a linear combination of inputs and weights, followed by sigmoid activation
    y = np.dot(x_random, weights)
    y = 1 / (1 + np.exp(-y))
    return x_random, y
    

def fit(x_random, y, error_tol):
    """
    Fits a model to the provided data using gradient descent to adjust weights and minimize the error.

    Parameters:
    - x_random : X values over n datapoints 
    - y : Target output values of polynomials using 'weights'
    - error_tol (float): Convergence tolerance for the error. The process stops when the error falls below this threshold.
    - y_mult: Updatin polynomial function values
    Returns:
    - error_all (float): Total error after convergence.
    """
    dimensions = len(x_random[0])
    num_func = len(y[0])

    learning_rate = 0.01
    weights_n = np.random.uniform(0, 2, (dimensions, num_func))
    fit.weight_updates = []
    fit.error_mat = []

    # Define the polynomial function as a dot product of inputs and weights
    def p1(x):
        return np.dot(x, weights_n)
    
    # Function to calculate the error as mean squared error
    def calculate_errors(z):
        return ((z)**2) / dimensions
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        return x * (1 - x)

    # Gradient descent loop
    for i in range(100000):
        # Compute predictions using sigmoid activation
        y_mult = sigmoid(p1(x_random))
        
        # Calculate the error and the gradient
        z = y_mult - y 
        error = calculate_errors(z)
        grad_a1 = np.dot(x_random.T, z * sigmoid_derivative(y_mult)) * learning_rate / dimensions
        #update the weights        
        weights_n -= grad_a1
        
        # Check if the error has fallen below the tolerance
        if np.all(error < error_tol):
            break
        
        # Store the updated weights and error for visualization
        fit.weight_updates.append(weights_n.copy())  # Store the updated weights
        fit.error_mat.append(np.sum(error))  # Store the sum of errors
    
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
    """
    Creates and displays an animation to visualize the weight updates and error convergence during training.

    The animation consists of two plots:
    - The left plot visualizes the changes in weight values over iterations. Initial weights are shown with a 
      blue gradient, and updated weights are shown with a red gradient.
    - The right plot shows the convergence of error values over iterations on a logarithmic scale.

    The animation updates both plots for each iteration to reflect changes in the model's performance.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Left plot for weight updates
    ax[0].set_title("Weight Updates")
    ax[0].set_xlabel("Weight Index")
    ax[0].set_ylabel("Weight Value")
    ax[0].set_xlim(0, len(fit.initial_weights.flatten()))
    ax[0].set_ylim(0, 5)
    
    # Create a colormap (blue gradient) for the initial weights
    initial_weights = fit.initial_weights.flatten()
    norm_initial = plt.Normalize(0, len(initial_weights))  # Normalize for colormap
    colors_initial = cm.Blues(norm_initial(range(len(initial_weights))))  # Apply blue gradient colormap
    
    # Create a colormap (red gradient) for the updated weights
    norm_update = plt.Normalize(0, len(initial_weights))  # Normalize for colormap
    colors_update = cm.Reds(norm_update(range(len(initial_weights))))  # Apply red gradient colormap
    
    # Scatter the initial weights using the blue gradient
    for idx in range(len(initial_weights)):
        ax[0].scatter(idx, initial_weights[idx], color=colors_initial[idx])
    
    # Initialize empty plot for current (changing) weights
    current_weights = ax[0].scatter([], [], color='red')

    # Right plot for error convergence
    ax[1].set_xlim(0, len(fit.error_mat))
    ax[1].set_ylim(min(fit.error_mat) * 0.1, max(fit.error_mat) * 10)
    ax[1].set_yscale('log')
    ax[1].set_title("Error Convergence")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Error Value")

    line_error, = ax[1].plot([], [], lw=2, color='green')

    # Update function for animation
    def update_animation(i):
        """
        Updates the animation for each frame.
        
        Parameters:
        - i : The current frame index.
        
        Returns:
        - current_weights: Updated scatter plot of current weights.
        - line_error: Updated line plot of error convergence.
        """
        # Update current weights in the left plot with red gradient
        ax[0].collections[-1].remove()  # Remove previous scatter plot
        current_weights = ax[0].scatter(range(len(initial_weights)), 
                                        fit.weight_updates[i].flatten(), 
                                        color=colors_update)
        
        # Update error convergence in the right plot
        line_error.set_data(range(i + 1), fit.error_mat[:i + 1])

        return current_weights, line_error,

    # Create the animation with blit=True for faster rendering
    anim = FuncAnimation(fig, update_animation, frames=len(fit.weight_updates), interval=100, blit=True, repeat=False)
    plt.show()




class tutorial():
    '''
    This class demonstrates how the `Func_fit` module functions by providing a tutorial on data creation, 
    model fitting, and visualization of the fitting process. 

    It includes methods to:
    1. Create synthetic data and target values.
    2. Fit a model to the generated data using gradient descent.
    3. Animate the fitting process to visualize how weights and error values evolve over iterations.

    The purpose of this class is to help users understand the mechanics of the `Func_fit` module, including
    how the data is generated, how the model is trained, and how the results are visualized. This can be 
    particularly useful for those who want to experiment with the module or adapt it for their own use cases.

    The class parameters include:
    - `dimensions`: Number of input features (dimensions) for the data.
    - `datapoints`: Number of data points to generate.
    - `num_func`: Number of output functions or target variables.
    - `error_tol`: Tolerance level for stopping criteria in the gradient descent process.
    - `learning_rate`: Learning rate used in gradient descent optimization.
    
    -weights is the final weight we are trying to approach
    -weights_n is the matrix that gets updated at each iteration and converges towards the final weights

    -x_random is the x over n datapoints generated randomly

    -y is the polyomial function matrix generated using 'weights' 
    -y_mult is the polynomial function matrix generated using 'weights_n'
    '''
    
    

    # Parameters for data generation and model training
    dimensions = 3  # Number of input features
    datapoints = 10  # Number of data points
    num_func = 2    # Number of output functions or target variables
    error_tol = 1e-4  # Error tolerance for stopping criteria in gradient descent
    learning_rate = 0.01  # Learning rate for gradient descent

    # List to store weight updates during training
    weight_updates = []
    # List to store error values during training
    error_mat = []
    
    @staticmethod
    def create_data():
        """
        Generates random input data and corresponding target values.
        
        Returns:
            x_random: numpy array of shape (datapoints, dimensions) containing random input values.
            y: numpy array of shape (datapoints, num_func) containing target values after applying a sigmoid function.
        """
        # Generate random input data in the range [-1, 1]
        x_random = np.random.uniform(-1, 1, (tutorial.datapoints, tutorial.dimensions))
        # Initialize random weights
        weights = np.random.uniform(0, 2, (tutorial.dimensions, tutorial.num_func))
        # Save initial weights
        tutorial.initial_weights = weights
        # Compute target values using a linear combination and apply sigmoid activation function
        y = np.dot(x_random, weights)
        y = 1 / (1 + np.exp(-y))  # Sigmoid activation
        return x_random, y

    @staticmethod
    def fit(x_random, y):
        """
        Fits a model to the provided data using gradient descent.
        
        Parameters:
            x_random: numpy array of shape (datapoints, dimensions) containing input data.
            y: numpy array of shape (datapoints, num_func) containing target values.
        
        Returns:
            weights_n: numpy array of shape (dimensions, num_func) containing the final weights after training.
        """
        # Initialize weights for the model
        weights_n = np.random.uniform(0, 2, (tutorial.dimensions, tutorial.num_func))

        # Define the polynomial function
        def p1(x):
            return np.dot(x, weights_n)

        # Define a function to calculate errors
        def calculate_errors(z):
            return ((z)**2) / tutorial.dimensions

        # Define sigmoid activation function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Define derivative of sigmoid function
        def sigmoid_derivative(x):
            return x * (1 - x)

        # Gradient descent loop
        for i in range(100000):
            # Compute predictions using the sigmoid function
            y_mult = sigmoid(p1(x_random))
            # Compute the difference between predictions and actual values
            z = y_mult - y
            # Calculate error
            error = calculate_errors(z)
            # Compute gradient
            grad_a1 = np.dot(x_random.T, z * sigmoid_derivative(y_mult)) * tutorial.learning_rate / tutorial.dimensions
            # Update weights
            weights_n -= grad_a1
            
            # Store updated weights and error
            tutorial.weight_updates.append(weights_n.copy())
            tutorial.error_mat.append(np.sum(error))
            
            # Check if error tolerance is met
            if np.all(error < tutorial.error_tol):
                break

        print(f"Converged in {i} iterations")
        return weights_n

    @staticmethod
    def animate():
        """
        Animates the weight updates and error convergence during the training process.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Left plot for weight updates
        ax[0].set_title("Weight Updates")
        ax[0].set_xlabel("Weight Index")
        ax[0].set_ylabel("Weight Value")
        ax[0].set_xlim(0, len(tutorial.initial_weights.flatten()))
        ax[0].set_ylim(0, 5)

        initial_weights = tutorial.initial_weights.flatten()
        current_weights, = ax[0].plot([], [], 'ro')
        ax[0].scatter(range(len(initial_weights)), initial_weights, color='blue')

        # Right plot for error convergence
        ax[1].set_xlim(0, len(tutorial.error_mat))
        ax[1].set_ylim(min(tutorial.error_mat) * 0.1, max(tutorial.error_mat) * 10)
        ax[1].set_yscale('log')
        ax[1].set_title("Error Convergence")
        ax[1].set_xlabel("Iteration")
        ax[1].set_ylabel("Error Value")

        line_error, = ax[1].plot([], [], lw=2, color='green')

        # Display the values of the weights
        weight_values = ax[0].text(0.5, 0.9, '', transform=ax[0].transAxes)

        def update_animation(i):
            # Update the plot with current weights
            current_weights.set_data(range(len(initial_weights)), tutorial.weight_updates[i].flatten())
            # Update the error convergence plot
            line_error.set_data(range(i + 1), tutorial.error_mat[:i + 1])
            # Update weight values display
            weight_values.set_text(f"Weights: {tutorial.weight_updates[i].flatten()}")

            return current_weights, line_error, weight_values

        anim = FuncAnimation(fig, update_animation, frames=len(tutorial.weight_updates), interval=100, blit=True, repeat=False)
        plt.show()

    @staticmethod
    def run():
        """
        Runs the entire process: creating data, fitting the model, and animating the results.
        """
        x_random, y = tutorial.create_data()
        tutorial.fit(x_random, y)
        tutorial.animate()


    

class playground():
    '''
    weights is the final weight we are trying to approach
    weights_n is the matrix that gets updated at each iteration and converges towards the final weights

    x_random is the x over n datapoints generated randomly

    y is the polyomial function matrix generated using 'weights' 
    y_mult is the polynomial function matrix generated using 'weights_n'
    '''
    @staticmethod
    def learning_rate(dimensions, num_func, datapoints):
        """
        Compares the effect of different learning rates on the convergence of gradient descent.
        
        Parameters:
            dimensions (int): Number of input features.
            num_func (int): Number of output functions or target variables.
            datapoints (int): Number of data points.
        
        This method visualizes error convergence for three different learning rate strategies:
        - Low Learning Rate
        - High Learning Rate
        - Adaptive Learning Rate (commented out)
        """
        err = 0.001
        x_random = np.random.uniform(-1, 1, (datapoints, dimensions))
        weights = np.random.uniform(0, 2, (dimensions, num_func))
        y = np.dot(x_random, weights)

        # Low learning rate gradient descent
        np.random.seed(4)
        weights_n = np.random.uniform(0, 2, (dimensions, num_func))
        
        def low():
            lr = 0.001  # Low learning rate
            weights_n_low = weights_n.copy()
            errors_low = []
            
            def p1(x):
                return np.dot(x, weights_n_low)
            
            def calculate_errors(z):
                return np.mean(z**2)  # Mean Squared Error (MSE)
            
            # Gradient Descent loop
            for i in range(100000):
                y_mult = p1(x_random)
                z = y_mult - y
                error = calculate_errors(z)
                errors_low.append(error)
                grad_a1 = np.dot(x_random.T, z) * lr / dimensions
                weights_n_low -= grad_a1
                if error < err:
                    break
            return errors_low

        # High learning rate gradient descent
        np.random.seed(4)
        weights_n = np.random.uniform(0, 2, (dimensions, num_func))
        
        def high():
            lr = 0.1  # High learning rate
            x_random_high = np.random.uniform(-1, 1, (datapoints, dimensions))
            y_high = np.dot(x_random_high, weights)
            weights_n_high = weights_n.copy()
            errors_high = []
            
            def p1(x):
                return np.dot(x, weights_n_high)
            
            def calculate_errors(z):
                return np.mean(z**2)  # Mean Squared Error (MSE)
            
            # Gradient Descent loop
            for i in range(100000):
                y_mult = p1(x_random_high)
                z = y_mult - y_high
                error = calculate_errors(z)
                errors_high.append(error)
                grad_a1 = np.dot(x_random_high.T, z) * lr / dimensions
                weights_n_high -= grad_a1
                if error < err:
                    break
            return errors_high

        '''
        # Adaptive Learning Rate Gradient Descent (commented out)
        def adap():
            lr = 0.00001
            epsilon = 1e-8
            x_random_adap = np.random.uniform(-1, 1, (datapoints, dimensions))
            y_adap = np.dot(x_random_adap, weights)
            weights_n_adap = weights_n
            errors_adap = []
            
            def p1(x):
                return np.dot(x, weights_n_adap)
            
            def calculate_errors(z):
                return np.mean(z**2)  # Mean Squared Error (MSE)
            
            grad_accumulator = np.zeros_like(weights_n_adap)
            # Gradient Descent loop with adaptive learning rate
            for i in range(100000):
                y_mult = p1(x_random_adap)
                z = y_mult - y_adap
                error = calculate_errors(z)
                errors_adap.append(error)
                grad_a1 = np.dot(x_random_adap.T, z) / dimensions
                grad_accumulator += grad_a1 ** 2
                adjusted_lr = lr / (np.sqrt(grad_accumulator) + epsilon)
                weights_n_adap -= adjusted_lr * grad_a1
                if error < err:
                    break
            
            errors_adap = np.zeros([5000, 100])
            return errors_adap
        '''

        # Run all gradient descent functions
        error_low = low()
        error_high = high()
        # error_adap = adap()  # Adaptive learning rate is commented out

        max_iter = max(len(error_low), len(error_high))
        # Function to pad error lists to the same length
        def pad_errors(errors):
            return errors + [errors[-1]] * (max_iter - len(errors))
        
        error_low = pad_errors(error_low)
        error_high = pad_errors(error_high)
        # error_adap = pad_errors(error_adap)  # For adaptive learning rate

        # Create plot to visualize error convergence
        fig, ax = plt.subplots()
        ax.set_xlim(0, max_iter)
        ax.set_ylim(-0.5, max(max(error_low), max(error_high)))

        low_line, = ax.plot([], [], label="Low Learning Rate", color="blue")
        high_line, = ax.plot([], [], label="High Learning Rate", color="green")
        # adap_line, = ax.plot([], [], label="Adap Learning Rate", color="red")

        def init():
            low_line.set_data([], [])
            high_line.set_data([], [])
            # adap_line.set_data([], [])
            return low_line, high_line

        def update(frame):
            x = np.arange(0, frame + 1)
            low_line.set_data(x, error_low[:frame + 1])
            high_line.set_data(x, error_high[:frame + 1])
            # adap_line.set_data(x, error_adap[:frame + 1])
            return low_line, high_line

        ani = FuncAnimation(fig, update, frames=max_iter, init_func=init, blit=True)
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Error")
        plt.title("Error Convergence for Different Learning Rates")
        plt.show()
    
    @staticmethod
    def error(dimensions, num_func, datapoints):
        """
        This method evaluates the performance of polynomial fitting using different error metrics: MAE, MSE, and infinity norm error.
        It visualizes the convergence of these errors over iterations.

        Parameters:
        dimensions (int): The number of dimensions for the polynomial.
        num_func (int): The number of output functions.
        datapoints (int): The number of data points to generate.

        Returns:
        None
        """
        # Generate random data and initialize weights
        x_random = np.random.uniform(-1, 1, (datapoints, dimensions))
        weights = np.random.uniform(0, 2, (dimensions, num_func))
        y = np.dot(x_random, weights)
        np.random.seed(4)
        weights_n = np.random.uniform(0, 2, (dimensions, num_func))
        
        def MAE(weights_n):
            """
            Compute and update weights using Mean Absolute Error (MAE) as the error metric.
            """
            y = np.dot(x_random, weights)
            lr = 0.1
            
            def p1(x):
                return np.dot(x, weights_n)
            
            def calculate_errors(z):
                return np.mean(np.abs(z))  # MAE
            
            errors = []
            for i in range(100000):
                y_mult = p1(x_random)
                z = y_mult - y
                error = calculate_errors(z)
                errors.append(error)
                grad_a1 = np.dot(x_random.T, z) * lr / dimensions
                weights_n -= grad_a1
                if error < 0.01:
                    break
            return errors

        def MSE(weights_n):
            """
            Compute and update weights using Mean Squared Error (MSE) as the error metric.
            """
            y = np.dot(x_random, weights)
            lr = 0.1
            
            def p1(x):
                return np.dot(x, weights_n)
            
            def calculate_errors(z):
                return np.mean(z**2)  # MSE
            
            errors = []
            for i in range(100000):
                y_mult = p1(x_random)
                z = y_mult - y
                error = calculate_errors(z)
                errors.append(error)
                grad_a1 = np.dot(x_random.T, z) * lr / dimensions
                weights_n -= grad_a1
                if error < 0.01:
                    break
            return errors
        
        def norm_inf(weights_n):
            """
            Compute and update weights using Infinity Norm (Max Error) as the error metric.
            """
            y = np.dot(x_random, weights)
            lr = 0.1
            
            def p1(x):
                return np.dot(x, weights_n)
            
            def calculate_errors(z):
                return np.max(np.abs(z))  # Infinity Norm Error
            
            errors = []
            for i in range(100000):
                y_mult = p1(x_random)
                z = y_mult - y
                error = calculate_errors(z)
                errors.append(error)
                grad_a1 = np.dot(x_random.T, z) * lr / dimensions
                weights_n -= grad_a1
                if error < 0.01:
                    break
            return errors
        
        # Compute errors
        mae_errors = MAE(weights_n)
        np.random.seed(4)
        weights_n = np.random.uniform(0, 2, (dimensions, num_func))
        mse_errors = MSE(weights_n)
        np.random.seed(4)
        weights_n = np.random.uniform(0, 2, (dimensions, num_func))
        inf_norm_errors = norm_inf(weights_n)
        
        # Determine the maximum number of iterations
        max_iterations = max(len(mae_errors), len(mse_errors), len(inf_norm_errors))
        
        # Pad the error lists to the same length
        def pad_errors(errors):
            return errors + [errors[-1]] * (max_iterations - len(errors))
        
        mae_errors = pad_errors(mae_errors)
        mse_errors = pad_errors(mse_errors)
        inf_norm_errors = pad_errors(inf_norm_errors)
        
        # Set up the plot
        fig, ax = plt.subplots()
        ax.set_xlim(0, max_iterations)
        ax.set_ylim(0, max(max(mae_errors), max(mse_errors), max(inf_norm_errors)))

        mae_line, = ax.plot([], [], label="MAE", color="blue")
        mse_line, = ax.plot([], [], label="MSE", color="green")
        inf_norm_line, = ax.plot([], [], label="Infinity Norm", color="red")
        
        def init():
            mae_line.set_data([], [])
            mse_line.set_data([], [])
            inf_norm_line.set_data([], [])
            return mae_line, mse_line, inf_norm_line

        def update(frame):
            x = np.arange(0, frame + 1)
            mae_line.set_data(x, mae_errors[:frame + 1])
            mse_line.set_data(x, mse_errors[:frame + 1])
            inf_norm_line.set_data(x, inf_norm_errors[:frame + 1])
            return mae_line, mse_line, inf_norm_line
        
        ani = FuncAnimation(fig, update, frames=max_iterations, init_func=init, blit=True)
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Error")
        plt.title("Error Convergence for MAE, MSE, and Infinity Norm")
        plt.show()

    @staticmethod
    def Grad_Desc(dimensions, num_func, datapoints):
        """
        Compares Gradient Descent (GD) and Stochastic Gradient Descent (SGD) with error convergence.
        
        Parameters:
            dimensions (int): Number of input features.
            num_func (int): Number of output functions or target variables.
            datapoints (int): Number of data points.
        
        This method visualizes error convergence for:
        - Gradient Descent (GD)
        - Stochastic Gradient Descent (SGD)
        """
        x_random = np.random.uniform(-1, 1, (datapoints, dimensions))
        weights = np.random.uniform(0, 2, (dimensions, num_func))
        y = np.dot(x_random, weights)
        np.random.seed(4)
        
        # Gradient Descent
        def GD():
            weights_n = np.random.uniform(0, 2, (dimensions, num_func))
            lr = 0.1
            error_list = []
            
            def p1(x):
                return np.dot(x, weights_n)
            
            def calculate_errors(z):
                return np.mean(np.abs(z))  # Mean Absolute Error (MAE)
            
            # Gradient Descent loop
            for i in range(1000):
                y_mult = p1(x_random)
                z = y_mult - y
                error = calculate_errors(z)
                grad_a1 = np.dot(x_random.T, z) * lr / dimensions
                weights_n -= grad_a1
                error_list.append(error)
                if error < 0.01:
                    break
            
            return error_list

        # Stochastic Gradient Descent
        def SGD():
            weights_n = np.random.uniform(0, 2, (dimensions, num_func))
            learning_rate = 0.1
            batch_size = 10
            num_iterations = 1000
            error_tol = 0.01
            error_list = []
            
            def p1(x):
                return np.dot(x, weights_n)
            
            def calculate_errors(z):
                return np.mean(z**2)  # Mean Squared Error (MSE)
            
            num_samples = x_random.shape[0]
            
            # Stochastic Gradient Descent loop
            #Choosing a random point and reducing its error and then repeating the same for other points and well
            for i in range(num_iterations):
                indices = np.random.permutation(num_samples)
                x_random_shuffled = x_random[indices]
                y_shuffled = y[indices]
                
                for start_idx in range(0, num_samples, batch_size):
                    end_idx = min(start_idx + batch_size, num_samples)
                    x_batch = x_random_shuffled[start_idx:end_idx]
                    y_batch = y_shuffled[start_idx:end_idx]
                    
                    y_mult = p1(x_batch)
                    z = y_mult - y_batch
                    error = calculate_errors(z)
                    grad_a1 = np.dot(x_batch.T, z) * learning_rate / batch_size
                    weights_n -= grad_a1
                
                error_list.append(error)
                if error < error_tol:
                    break
            
            return error_list

        # Run both GD and SGD functions
        gd_errors = GD()
        sgd_errors = SGD()

        # Determine the maximum number of iterations
        max_iterations = max(len(gd_errors), len(sgd_errors))

        # Function to pad error lists to the same length
        def pad_errors(errors):
            return errors + [errors[-1]] * (max_iterations - len(errors))

        gd_errors = pad_errors(gd_errors)
        sgd_errors = pad_errors(sgd_errors)

        # Create plot to visualize error convergence
        fig, ax = plt.subplots()
        ax.set_xlim(0, max_iterations)
        ax.set_ylim(0, max(max(gd_errors), max(sgd_errors)))

        gd_line, = ax.plot([], [], label="GD", color="blue")
        sgd_line, = ax.plot([], [], label="SGD", color="green")

        def init():
            gd_line.set_data([], [])
            sgd_line.set_data([], [])
            return gd_line, sgd_line

        def update(frame):
            x = np.arange(0, frame + 1)
            gd_line.set_data(x, gd_errors[:frame + 1])
            sgd_line.set_data(x, sgd_errors[:frame + 1])
            return gd_line, sgd_line

        ani = FuncAnimation(fig, update, frames=max_iterations, init_func=init, blit=True)
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Error")
        plt.title("Error Convergence for GD and SGD")
        plt.show()

    @staticmethod
    def var_inp(dimensions, num_func, datapoints):
        """
        Compares the effect of varying input sizes on gradient descent convergence.
        
        Parameters:
            dimensions (int): Number of input features.
            num_func (int): Number of output functions or target variables.
            datapoints (int): Number of data points.
        
        This method visualizes error convergence for different input sizes:
        - Original Input Size
        - Input Size 10x Original
        - Input Size Half of Original
        """
        
        # Gradient Descent for original input size
        def original():
            x_random = np.random.uniform(-1, 1, (datapoints, dimensions))
            weights = np.random.uniform(0, 2, (dimensions, num_func))
            y = np.dot(x_random, weights)
            np.random.seed(4)
            weights_n = np.random.uniform(0, 2, (dimensions, num_func))
            lr = 0.1
            error_list = []

            def p1(x):
                return np.dot(x, weights_n)

            def calculate_errors(z):
                return np.mean(np.abs(z))  # Mean Absolute Error (MAE)

            # Gradient Descent loop
            for i in range(1000):
                y_mult = p1(x_random)
                z = y_mult - y
                error = calculate_errors(z)
                grad_a1 = np.dot(x_random.T, z) * lr / dimensions
                weights_n -= grad_a1
                error_list.append(error)
                if error < 0.01:
                    break

            return error_list

        # Gradient Descent for input size 10 times original
        def ten_time():
            new_datapoints = datapoints * 10
            x_random = np.random.uniform(-1, 1, (new_datapoints, dimensions))
            weights = np.random.uniform(0, 2, (dimensions, num_func))
            y = np.dot(x_random, weights)
            np.random.seed(4)
            weights_n = np.random.uniform(0, 2, (dimensions, num_func))
            lr = 0.1
            error_list = []

            def p1(x):
                return np.dot(x, weights_n)

            def calculate_errors(z):
                return np.mean(np.abs(z))  # Mean Absolute Error (MAE)

            # Gradient Descent loop
            for i in range(1000):
                y_mult = p1(x_random)
                z = y_mult - y
                error = calculate_errors(z)
                grad_a1 = np.dot(x_random.T, z) * lr / dimensions
                weights_n -= grad_a1
                error_list.append(error)
                if error < 0.01:
                    break

            return error_list

        # Gradient Descent for input size half of original
        def half():
            if datapoints % 2 == 0:
                new_datapoints = datapoints // 2
            else:
                new_datapoints = (datapoints + 1) // 2
            x_random = np.random.uniform(-1, 1, (new_datapoints, dimensions))
            weights = np.random.uniform(0, 2, (dimensions, num_func))
            y = np.dot(x_random, weights)
            np.random.seed(4)
            weights_n = np.random.uniform(0, 2, (dimensions, num_func))
            lr = 0.1
            error_list = []

            def p1(x):
                return np.dot(x, weights_n)

            def calculate_errors(z):
                return np.mean(np.abs(z))  # Mean Absolute Error (MAE)

            # Gradient Descent loop
            for i in range(1000):
                y_mult = p1(x_random)
                z = y_mult - y
                error = calculate_errors(z)
                grad_a1 = np.dot(x_random.T, z) * lr / dimensions
                weights_n -= grad_a1
                error_list.append(error)
                if error < 0.01:
                    break

            return error_list

        # Run Gradient Descent for different input sizes
        original_errors = original()
        ten_time_errors = ten_time()
        half_errors = half()

        # Determine the maximum number of iterations
        max_iterations = max(len(original_errors), len(ten_time_errors), len(half_errors))

        # Function to pad error lists to the same length
        def pad_errors(errors):
            return errors + [errors[-1]] * (max_iterations - len(errors))

        original_errors = pad_errors(original_errors)
        ten_time_errors = pad_errors(ten_time_errors)
        half_errors = pad_errors(half_errors)

        # Create plot to visualize error convergence
        fig, ax = plt.subplots()
        ax.set_xlim(0, max_iterations)
        ax.set_ylim(0, max(max(original_errors), max(ten_time_errors), max(half_errors)))

        original_line, = ax.plot([], [], label="Original", color="blue")
        ten_time_line, = ax.plot([], [], label="10x Input", color="green")
        half_line, = ax.plot([], [], label="Half Input", color="red")

        def init():
            original_line.set_data([], [])
            ten_time_line.set_data([], [])
            half_line.set_data([], [])
            return original_line, ten_time_line, half_line

        def update(frame):
            x = np.arange(0, frame + 1)
            original_line.set_data(x, original_errors[:frame + 1])
            ten_time_line.set_data(x, ten_time_errors[:frame + 1])
            half_line.set_data(x, half_errors[:frame + 1])
            return original_line, ten_time_line, half_line

        ani = FuncAnimation(fig, update, frames=max_iterations, init_func=init, blit=True)
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Error")
        plt.title("Error Convergence for Different Input Sizes")
        plt.show()

            

            




    
