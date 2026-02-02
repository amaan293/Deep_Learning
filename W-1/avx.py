import numpy as np
import matplotlib.pyplot as plt

def initialize_weights(n_features):
    np.random.seed(42)
    w = np.random.uniform(-1, 1, n_features)
    b = np.random.uniform(-1, 1)
    return w, b

def perceptron_train(X, y, lr=0.1, epochs=50):
    w, b = initialize_weights(X.shape[1])
    print("Initial Weights:", w)
    print("Initial Bias:", b)
    print("\nTraining started-->")
    for _ in range(epochs):
        for i in range(len(X)):
            linear_output = np.dot(X[i], w) + b
            y_pred = 1 if linear_output >= 0 else -1
            if y[i] != y_pred:
                w += lr * y[i] * X[i]
                b += lr * y[i]
    print("\nTraining completed")
    print("Final Weights:", w)
    print("Final Bias:", b)
    return w, b

def perceptron_predict(X, w, b):
    return np.array([1 if np.dot(x, w) + b >= 0 else -1 for x in X])

X = np.array([
    [2, 3],
    [4, 5],
    [5, 10],
    [1, 4],
    [3, 4],
    [6, 2],
    [7, 3],
    [8, 1],
    [2, 8],
    [3, 7],
    [5, 6],
    [6, 5],
    [3, 6],
    [5, 1],
    [4, 10],
    [3, 7],
    [6, 10]
])

y = np.array([-1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1])

w, b = perceptron_train(X, y, lr=0.1, epochs=20)

def plot_decision_boundary(X, y, w, b):
    plt.scatter(X[y==1][:,0], X[y==1][:,1], color='blue', label='Class +1')
    plt.scatter(X[y==-1][:,0], X[y==-1][:,1], color='red', label='Class -1')
    x_vals = np.linspace(0, 9, 100)
    y_vals = -(w[0] * x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.7)
    plt.axvline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.7)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.title("Perceptron Decision Boundary")
    plt.grid(True, alpha=0.3)
    plt.show()

plot_decision_boundary(X, y, w, b)

def activation_step_function(weighted_sum):
    return 1 if weighted_sum >= 0 else 0

def compute_weighted_sum(inputs, weights, bias):
    return np.dot(inputs, weights) + bias

def xor_using_perceptron_network(input_pair):
    or_weights = [1, 1]
    or_bias = -0.5
    or_gate_output = activation_step_function(compute_weighted_sum(input_pair, or_weights, or_bias))
    nand_weights = [-1, -1]
    nand_bias = 1.5
    nand_gate_output = activation_step_function(compute_weighted_sum(input_pair, nand_weights, nand_bias))
    and_weights = [1, 1]
    and_bias = -1.5
    combined_inputs = [or_gate_output, nand_gate_output]
    xor_output = activation_step_function(compute_weighted_sum(combined_inputs, and_weights, and_bias))
    return xor_output

X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

print("x1 | x2 | XOR")
for input_values in X_xor:
    xor_result = xor_using_perceptron_network(input_values)
    print(f"   {input_values[0]}        {input_values[1]}        {xor_result}")

grid_resolution = 200
grid_range = (-0.2, 1.2)

x1_grid, x2_grid = np.meshgrid(
    np.linspace(grid_range[0], grid_range[1], grid_resolution),
    np.linspace(grid_range[0], grid_range[1], grid_resolution)
)

grid_predictions = np.array([
    xor_using_perceptron_network([x1, x2]) 
    for x1, x2 in zip(x1_grid.ravel(), x2_grid.ravel())
])
grid_predictions = grid_predictions.reshape(x1_grid.shape)

plt.contourf(x1_grid, x2_grid, grid_predictions, alpha=0.4, cmap='RdYlBu')

xor_inputs = [0, 0, 1, 1]
xor_outputs = [0, 1, 0, 1]
point_colors = ['red', 'blue', 'blue', 'red']

plt.scatter(xor_inputs, xor_outputs, 
           c=point_colors, s=100, edgecolors='black', linewidths=2)

plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.title("XOR Decision Regions Using Perceptron Network")
plt.grid(True, alpha=0.3)
plt.show()
