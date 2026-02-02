import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, eta=0.01, max_iters=1000):
        self.eta = eta
        self.max_iters = max_iters

    def sign(self, z):
        return 1 if z >= 0 else -1

    def train(self, X, y):
        m, n = X.shape
        self.w = np.random.randn(n) * 0.01
        self.b = 0.0

        for _ in range(self.max_iters):
            errors = 0
            for i in range(m):
                z = np.dot(X[i], self.w) + self.b
                y_hat = self.sign(z)

                if y[i] * y_hat <= 0:
                    self.w += self.eta * y[i] * X[i]
                    self.b += self.eta * y[i]
                    errors += 1

            if errors == 0:
                break

    def predict(self, X):
        return np.array([self.sign(np.dot(x, self.w) + self.b) for x in X])

np.random.seed(0)

X_pos = np.random.randn(50, 2) + [2, 2]
y_pos = np.ones(50)

X_neg = np.random.randn(50, 2) + [-2, -2]
y_neg = -np.ones(50)

X_train = np.vstack((X_pos, X_neg))
y_train = np.hstack((y_pos, y_neg))

model = Perceptron()
model.train(X_train, y_train)

# Training accuracy
y_train_pred = model.predict(X_train)
train_acc = np.mean(y_train_pred == y_train)

print("=== PERCEPTRON RESULTS ===")
print("Weights:", model.w)
print("Bias:", model.b)
print(f"Training Accuracy: {train_acc*100:.2f}%")

X_test_pos = np.random.randn(40, 2) + [1.2, 1.2]
y_test_pos = np.ones(40)

X_test_neg = np.random.randn(40, 2) + [-1.2, -1.2]
y_test_neg = -np.ones(40)

X_test = np.vstack((X_test_pos, X_test_neg))
y_test = np.hstack((y_test_pos, y_test_neg))

y_test_pred = model.predict(X_test)
test_acc = np.mean(y_test_pred == y_test)

print(f"Test Accuracy: {test_acc*100:.2f}%")

plt.figure(figsize=(7,5))
plt.scatter(X_pos[:,0], X_pos[:,1], c="blue", label="+1")
plt.scatter(X_neg[:,0], X_neg[:,1], c="red", label="-1")

x_vals = np.linspace(-5, 5, 100)
y_vals = -(model.w[0]*x_vals + model.b) / model.w[1]
plt.plot(x_vals, y_vals, "k--", label="Decision Boundary")

plt.title("Perceptron Decision Boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid()
plt.show()

def step(z):
    return 1 if z >= 0 else 0

def perceptron_gate(x, w, b):
    return step(w[0]*x[0] + w[1]*x[1] + b)

# XOR dataset
X_xor = [(0,0), (0,1), (1,0), (1,1)]
y_xor = [0, 1, 1, 0]

# AND
w_and, b_and = [1,1], -1.5
# NAND
w_nand, b_nand = [-1,-1], 1.5
# OR
w_or, b_or = [1,1], -0.5

print("\n=== XOR RESULTS ===")
for x in X_xor:
    h1 = perceptron_gate(x, w_or, b_or)
    h2 = perceptron_gate(x, w_nand, b_nand)
    y = perceptron_gate((h1, h2), w_and, b_and)
    print(f"{x} -> {y}")


colors = ["red" if y == 0 else "blue" for y in y_xor]

plt.figure(figsize=(5,5))
plt.scatter([x[0] for x in X_xor], [x[1] for x in X_xor], c=colors)
plt.title("XOR Data (Not Linearly Separable)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid()
plt.show()
