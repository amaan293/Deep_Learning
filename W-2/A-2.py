import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

BOOLEAN_FUNCTIONS = {
    "AND":   [0, 0, 0, 1],
    "OR":    [0, 1, 1, 1],
    "XOR":   [0, 1, 1, 0],
}

HIDDEN_OUTPUTS = np.eye(4)

def step(z):
    return 1 if z >= 0 else 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_threshold(truth_table, epochs=10):
    w = np.random.uniform(-1, 1, 4)
    b = np.random.uniform(-1, 1)
    y = np.array(truth_table)
    
    print("\nInitial weights:", w)
    print("Initial bias:", b)
    
    for epoch in range(epochs):
        errors = 0
        for i in range(4):
            z = np.dot(w, HIDDEN_OUTPUTS[i]) + b 
            y_hat = step(z)
            
            if y_hat != y[i]:
                errors += 1
                w += (y[i] - y_hat) * HIDDEN_OUTPUTS[i]
                b += (y[i] - y_hat)
        
        if errors == 0:
            print(f"Converged at epoch {epoch + 1}")
            break
    print(w)
    print(b)
    return w, b

def predict_threshold(x, w, b):
    z = np.dot(w, x) + b
    return step(z)

def train_sigmoid(truth_table, epochs=10):
    w = np.random.uniform(-1, 1, 4)
    b = np.random.uniform(-1, 1)
    y = np.array(truth_table)
    
    print("\nInitial weights:", w)
    print("Initial bias:", b)
    
    for epoch in range(epochs):
        errors = 0
        for i in range(4):
            z = np.dot(w, HIDDEN_OUTPUTS[i]) + b
            y_hat = sigmoid(z)
            error = y[i] - y_hat
            
            pred = 1 if y_hat >= 0.5 else 0
            if pred != y[i]:
                errors += 1
            
            w += error * HIDDEN_OUTPUTS[i]
            b += error
        
        if errors == 0:
            print(f"Converged at epoch {epoch + 1}")
            break
    print(w)
    print(b)
    return w, b

def predict_sigmoid(x, w, b):
    z = np.dot(w, x) + b
    activation = sigmoid(z)
    return 1 if activation >= 0.5 else 0

def test_function(name, truth_table, w_th, b_th, w_sg, b_sg):
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    
    print("\n[Threshold Activation]")
    print("x1 x2 | Target | Prediction")
    correct_th = 0
    for i in range(4):
        pred = predict_threshold(HIDDEN_OUTPUTS[i], w_th, b_th)
        print(f" {X[i][0]}  {X[i][1]}  |   {truth_table[i]}    |     {pred}")
        if pred == truth_table[i]:
            correct_th += 1
    print(f"Accuracy: {(correct_th/4)*100:.2f}%")
    
    print("\n[Sigmoid Activation]")
    print("x1 x2 | Target | Prediction")
    correct_sg = 0
    for i in range(4):
        pred = predict_sigmoid(HIDDEN_OUTPUTS[i], w_sg, b_sg)
        print(f" {X[i][0]}  {X[i][1]}  |   {truth_table[i]}    |     {pred}")
        if pred == truth_table[i]:
            correct_sg += 1
    print(f"Accuracy: {(correct_sg/4)*100:.2f}%")

for name, truth_table in BOOLEAN_FUNCTIONS.items():
    
    print(f"Training {name}")

    
    print("\n--- THRESHOLD ACTIVATION ---")
    w_th, b_th = train_threshold(truth_table, epochs=10)
    print(w_th)
    print("\n--- SIGMOID ACTIVATION ---")
    w_sg, b_sg = train_sigmoid(truth_table, epochs=10)
    
    test_function(name, truth_table, w_th, b_th, w_sg, b_sg)
    
