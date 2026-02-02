import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

X0 = np.random.randn(50, 2) + np.array([1, 1])
y0 = np.zeros(50)

X1 = np.random.randn(50, 2) + np.array([5,5])
y1 = np.ones(50)

X = np.vstack((X0, X1))
y = np.hstack((y0, y1))
print(X.shape)
print(y.shape)

def step(z):
    return 1 if z >= 0 else 0

def train_perceptron(X, y, lr= 0.01, epochs=30):
    w = np.zeros(X.shape[1])
    b = 0

    print("P1: Binary Classification")
    print("Initial weights:", w)
    print("Initial bias:", b)

    for epoch in range(epochs):
        if epoch == 0:
            print("\nEpoch 1 updates:")
        for i in range(len(X)):
            z = np.dot(w, X[i]) + b
            y_hat = step(z)
            err = y[i] - y_hat

            if epoch == 0 and err != 0:
                print(f" Sample {i}")
                print("  Input:", X[i])
                print("  True:", y[i], "Pred:", y_hat)
                print("  Error:", err)
                print("  w_old:", w, "b_old:", b)

            w = w + lr * err * X[i]
            b = b + lr * err

            if epoch == 0 and err != 0:
                print("  w_new:", w, "b_new:", b)

    print("\nFinal weights:", w)
    print("Final bias:", b)

    return w, b

def predict(X, w, b):
    y_pred = []
    for i in range(len(X)):
        z = np.dot(w, X[i]) + b
        y_pred.append(step(z))
    return np.array(y_pred)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

w, b = train_perceptron(X, y)
y_pred = predict(X, w, b)
acc = accuracy(y, y_pred)
print("\nAccuracy:", acc)

plt.scatter(X[:, 0], X[:, 1], c=y)
x_vals = np.array([X[:, 0].min(), X[:, 0].max()])
y_vals = -(w[0] * x_vals + b) / w[1]
plt.plot(x_vals, y_vals, 'k-')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("P1: Perceptron Decision Boundary")
plt.show()

X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])

y_or = np.array([0,1,1,1])
w_or, b_or = train_perceptron(X_xor, y_or)

y_nand = np.array([1,1,1,0])
w_nand, b_nand = train_perceptron(X_xor, y_nand)

w_out = np.array([1,1])
b_out = -1.5

outputs = []
print("\nP2: XOR Network using OR + NAND -> AND")
for i in range(len(X_xor)):
    x = X_xor[i]
    h1 = step(np.dot(w_or, x) + b_or)       # OR
    h2 = step(np.dot(w_nand, x) + b_nand)   # NAND
    out = step(np.dot(w_out, np.array([h1,h2])) + b_out)  # AND
    outputs.append(out)
    print(f"Input: {x} -> OR: {h1}, NAND: {h2}, XOR: {out}")

outputs = np.array(outputs)
acc_xor = np.sum(outputs == y_xor)/len(y_xor)
print("XOR Accuracy:", acc_xor)

colors = ['purple' if o==0 else 'cyan' for o in outputs]
plt.figure(figsize=(5,5))
plt.scatter(X_xor[:,0], X_xor[:,1], c=colors, s=200)
for i, txt in enumerate(outputs):
    plt.annotate(txt, (X_xor[i,0]+0.05, X_xor[i,1]+0.05), fontsize=12)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("P2: XOR Network Output")
plt.grid(True)
plt.show()

xx, yy = np.meshgrid(np.linspace(-0.2,1.2,200), np.linspace(-0.2,1.2,200))
Z = np.zeros_like(xx)

for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        x_grid = np.array([xx[i,j], yy[i,j]])
        h1 = step(np.dot(w_or, x_grid) + b_or)
        h2 = step(np.dot(w_nand, x_grid) + b_nand)
        Z[i,j] = step(np.dot(w_out, np.array([h1,h2])) + b_out)

plt.figure(figsize=(5,5))
plt.contourf(xx, yy, Z, alpha=0.3,
             levels=[-0.1,0.5,1.1], colors=['purple','cyan'])
plt.scatter(X_xor[:,0], X_xor[:,1],
            c=['purple' if o==0 else 'cyan' for o in outputs], s=200)
for i, txt in enumerate(outputs):
    plt.annotate(txt, (X_xor[i,0]+0.02, X_xor[i,1]+0.02), fontsize=12)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("P2: XOR Decision Region (OR + NAND -> AND)")
plt.grid(True)
plt.show()
