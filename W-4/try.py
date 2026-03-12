import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


X_train, y_train = make_moons(n_samples=1500, noise=0.15, random_state=42)
X_test,  y_test  = make_moons(n_samples=500,  noise=0.15, random_state=43)

scaler  = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

y_train = y_train.reshape(-1, 1)
y_test  = y_test.reshape(-1, 1)

print("Dataset ready  |  Train: 1500 samples  |  Test: 500 samples")


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh_fn(z):
    return np.tanh(z)

def tanh_deriv(z):
    return 1 - np.tanh(z) ** 2

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_deriv(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)

ACTIVATIONS = {
    "Sigmoid":    (sigmoid,    sigmoid_deriv),
    "Tanh":       (tanh_fn,    tanh_deriv),
    "ReLU":       (relu,       relu_deriv),
    "Leaky ReLU": (leaky_relu, leaky_relu_deriv),
}


def init_weights(n_in, n_hidden, n_out, seed=42):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((n_in, n_hidden))  * np.sqrt(2 / n_in)
    b1 = np.zeros((1, n_hidden))
    W2 = rng.standard_normal((n_hidden, n_out)) * np.sqrt(2 / n_hidden)
    b2 = np.zeros((1, n_out))
    return W1, b1, W2, b2


def forward(X, W1, b1, W2, b2, act_fn):
    Z1 = X @ W1 + b1
    A1 = act_fn(Z1)
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2


def bce_loss(y_true, y_pred):
    eps = 1e-12
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))


def train(X, y, act_name, n_hidden=48, lr=0.01, n_iter=10000, batch_size=64, log_every=500, seed=42):
    act_fn, act_d = ACTIVATIONS[act_name]
    W1, b1, W2, b2 = init_weights(X.shape[1], n_hidden, 1, seed=seed)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    loss_history = []

    for it in range(1, n_iter + 1):
        idx = rng.choice(n, batch_size, replace=False)
        Xb, yb = X[idx], y[idx]

        Z1, A1, Z2, A2 = forward(Xb, W1, b1, W2, b2, act_fn)

        dA2 = (A2 - yb) / batch_size
        dZ2 = dA2 * sigmoid_deriv(Z2)
        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * act_d(Z1)
        dW1 = Xb.T @ dZ1
        db1 = dZ1.sum(axis=0, keepdims=True)

        W1 -= lr * dW1;  b1 -= lr * db1
        W2 -= lr * dW2;  b2 -= lr * db2

        if it % log_every == 0:
            _, _, _, A2_full = forward(X, W1, b1, W2, b2, act_fn)
            loss = bce_loss(y, A2_full)
            loss_history.append((it, loss))
            print(f"  iter {it:>5}  |  loss: {loss:.4f}")

    return W1, b1, W2, b2, loss_history


results = {}
NAMES = ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU"]

for name in NAMES:
    print(f"\nTraining [{name}]  --  2 -> 48 -> 1  |  lr=0.01  |  batch=64  |  10000 iters")
    W1, b1, W2, b2, lh = train(X_train, y_train, name)

    act_fn = ACTIVATIONS[name][0]
    _, _, _, A2_tr = forward(X_train, W1, b1, W2, b2, act_fn)
    _, _, _, A2_te = forward(X_test,  W1, b1, W2, b2, act_fn)

    train_acc  = np.mean((A2_tr >= 0.5).astype(int) == y_train) * 100
    test_acc   = np.mean((A2_te >= 0.5).astype(int) == y_test)  * 100
    final_loss = lh[-1][1]

    results[name] = dict(W1=W1, b1=b1, W2=W2, b2=b2,
                         loss_history=lh,
                         train_acc=train_acc,
                         test_acc=test_acc,
                         final_loss=final_loss)
    print(f"  Done  |  train acc: {train_acc:.2f}%  |  test acc: {test_acc:.2f}%")


print("\n")
print(f"  {'Activation':<14} {'Architecture':<13} {'Final Loss':<13} {'Train Acc':<12} {'Test Acc'}")
print(f"  {'-'*14} {'-'*13} {'-'*13} {'-'*12} {'-'*9}")
for name in NAMES:
    r = results[name]
    print(f"  {name:<14} {'2->48->1':<13} {r['final_loss']:<13.4f} {str(round(r['train_acc'],2))+'%':<12} {r['test_acc']:.2f}%")
print()


margin = 0.5
x_min = min(X_train[:,0].min(), X_test[:,0].min()) - margin
x_max = max(X_train[:,0].max(), X_test[:,0].max()) + margin
y_min = min(X_train[:,1].min(), X_test[:,1].min()) - margin
y_max = max(X_train[:,1].max(), X_test[:,1].max()) + margin
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

def predict_grid(xx, yy, r, act_fn):
    G  = np.c_[xx.ravel(), yy.ravel()]
    Z1 = G @ r["W1"] + r["b1"]
    A1 = act_fn(Z1)
    Z2 = A1 @ r["W2"] + r["b2"]
    return sigmoid(Z2).reshape(xx.shape)


fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Neural Network -- 2 inputs | 48 hidden | 1 output | BCE loss | 10000 iters", fontsize=13)

COLORS = ["blue", "green", "red", "orange"]

for i, name in enumerate(NAMES):
    r = results[name]
    lh = r["loss_history"]
    iters, losses = zip(*lh)

    ax_loss = axes[0][i]
    ax_loss.plot(iters, losses, color=COLORS[i], linewidth=2)
    ax_loss.set_title(f"{name} -- Loss Curve")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("BCE Loss")
    ax_loss.grid(True)

    ax_db = axes[1][i]
    Z = predict_grid(xx, yy, r, ACTIVATIONS[name][0])
    ax_db.contourf(xx, yy, Z, levels=50, cmap="RdBu_r", alpha=0.8)
    ax_db.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5)

    ax_db.scatter(X_test[y_test.ravel()==0, 0], X_test[y_test.ravel()==0, 1],
                  s=15, color="blue", alpha=0.7, label="Class 0")
    ax_db.scatter(X_test[y_test.ravel()==1, 0], X_test[y_test.ravel()==1, 1],
                  s=15, color="red", alpha=0.7, label="Class 1")

    ax_db.set_title(f"{name} -- Test Acc: {r['test_acc']:.2f}%")
    ax_db.set_xlabel("Feature 1")
    ax_db.set_ylabel("Feature 2")
    ax_db.legend(fontsize=8)

plt.tight_layout()
plt.show()