import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx

# Neural network weight visualization with input values and weight values displayed
def plot_neural_network(weights_n, initial_weights, inputs, taylor_terms, title='Neural Network Architecture'):
    # Define layers
    n_input = weights_n.shape[0]  # Number of input nodes based on weights shape
    n_output = 1  # Single output node for sine prediction

    # Generate graph
    G = nx.DiGraph()

    input_nodes = [f'Input_{i + 1}' for i in range(n_input)]
    output_node = ['Output']

    # Node positions
    pos = {}
    for i, node in enumerate(input_nodes):
        pos[node] = (0, i)  # Inputs on the left side
    
    pos['Output'] = (1, n_input // 2)  # Output on the right side

    # Minimum thickness for edges
    min_thickness = 1

    # Add edges between input and output, and vary edge thickness based on weights
    for i, node in enumerate(input_nodes):
        weight = weights_n[i, 0]
        # Ensure thickness is at least the minimum value
        thickness = max(min_thickness, weight * 10)
        G.add_edge(node, 'Output', weight=thickness)

    # Plot the graph
    plt.figure(figsize=(10, 8))
    edges = G.edges(data=True)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color='lightblue', node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=output_node, node_color='lightgreen', node_size=800)

    # Draw edges with varying thickness based on weights
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=[d['weight'] for (u, v, d) in edges], edge_color='blue')

    # Add labels to nodes (input values, weights, and Taylor series terms)
    for i, node in enumerate(input_nodes):
        input_value = inputs[:, i].mean()  # Use mean input value for the node
        weight_value = weights_n[i, 0]
        term_description = taylor_terms[i] if i < len(taylor_terms) else 'N/A'  # Protect against index error
        nx.draw_networkx_labels(G, pos, labels={node: f'{node}\nInput: {input_value:4f}\nWeight: {weight_value:.4f}\nTerm: {term_description}'},font_size=6)
    
    # Add label for the output node
    nx.draw_networkx_labels(G, pos, labels={'Output': 'Output'})

    plt.title(title)
    plt.axis('off')
    plt.show()

# Your sine function (as is, with Adam optimizer)
def sine(dimensions, num_func, datapoints):
    x_orr = np.random.uniform(-2.5, 2.5, datapoints)
    x_random = np.matrix([(x_orr**i) for i in range(0, 2*dimensions)]).T    
    weights = np.array([(1 if i % 4 == 1 else -1) / math.factorial(i) if i % 2 == 1 else 0 for i in range(2 * dimensions)]).reshape(-1, 1)
    print(x_random)
    print(weights)
    y_target = np.dot(x_random, weights)
    y = np.array(np.sin(x_orr))   
    weights_n = np.array([(1 if i % 4 == 1 else -1) / math.factorial(i) if i % 2 == 1 else 0 for i in range(2 * dimensions)]).reshape(-1, 1)
    weights_n=weights_n*0.7
    print(weights_n)
    # Capture Taylor series terms for each input
    taylor_terms = [f'x^{i-1}' for i in range(1, 2*dimensions)]

    def p1(x, weights_n):
        return np.dot(x, weights_n)
    
    def calculate_errors(z):
        return np.square(z)/datapoints   

    learning_rate = 0.001  
    beta1 = 0.9  
    beta2 = 0.999  
    epsilon = 1e-8      
    m = np.zeros(weights_n.shape)  
    v = np.zeros(weights_n.shape)  
   
    initial_weights = weights_n.copy()  # Save initial weights for plotting

    for i in range(10000):
        y_mult = p1(x_random, weights_n)
        z = y_mult - y_target
        error = calculate_errors(z) 
        grad = np.dot(z.T, x_random).T / datapoints               

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * np.square(grad)       
        m_hat = m / (1 - beta1**(i+1))
        v_hat = v / (1 - beta2**(i+1))      
        weights_n -= (learning_rate * m_hat) / (np.sqrt(v_hat) + epsilon)

        if np.sum(error) < 0.00001:
            break

    # Plot neural network architecture with weights after training
    plot_neural_network(weights_n, initial_weights, x_random, taylor_terms, title='Neural Network with Final Weights and Inputs')
    
    return error, y_mult, x_orr, y_target, i, initial_weights, weights_n

# Run sine function and plot results
error, y_new, x, y_orig, iterations, initial_weights, final_weights = sine(6, 1, 300)
print(f"Iterations: {iterations}")

x = np.asarray(x).reshape(-1)  
y_orig = np.asarray(y_orig).reshape(-1)  
y_new = np.asarray(y_new).reshape(-1) 

# Plot original vs predicted sine wave
plt.figure()
plt.scatter(x, y_orig, label='Original')
plt.scatter(x, y_new, label='Predicted')
plt.legend()
plt.title("Original vs Predicted using Adam Optimizer")
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()
