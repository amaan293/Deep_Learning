import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def load_data():
    feature_names = ["MedInc", "AveRooms"]

    try:
        data = fetch_california_housing()
        feature_indices = [0, 2]

        X_raw = data.data[:, feature_indices]   
        y_raw = data.target                    

        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_raw), size=500, replace=False)
        X_raw, y_raw = X_raw[idx], y_raw[idx]
        print("[INFO] Loaded real California Housing dataset.")

    except Exception as e:
        print(f"[WARN] Could not load California Housing ({e}).")
        print("[INFO] Using synthetic dataset with same feature distributions.")

        rng = np.random.default_rng(42)
        n = 500
        MedInc = rng.gamma(shape=3.0, scale=1.5, size=n)
        AveRooms = rng.normal(loc=5.5, scale=1.5, size=n).clip(2.0, 14.0)

        MedHouseVal = (0.7 * MedInc
                       + 0.15 * AveRooms
                       + rng.normal(0, 0.5, n)).clip(0.5, 5.0)

        X_raw = np.column_stack([MedInc, AveRooms])
        y_raw = MedHouseVal

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X_raw)
    y = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

    return X, y, feature_names

X_DATA, Y_DATA, FEATURE_NAMES = load_data()

INPUT_SIZE  = 2   
HIDDEN_SIZE = 6    
OUTPUT_SIZE = 1    

class NeuralNetwork:

    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)

        self.W1 = rng.normal(0, np.sqrt(2 / (INPUT_SIZE + HIDDEN_SIZE)),
                             (INPUT_SIZE, HIDDEN_SIZE))   # (2, 6)
        self.b1 = np.zeros(HIDDEN_SIZE)                   # (6,)
        self.W2 = rng.normal(0, np.sqrt(2 / (HIDDEN_SIZE + OUTPUT_SIZE)),
                             (HIDDEN_SIZE, OUTPUT_SIZE))  # (6, 1)
        self.b2 = np.zeros(OUTPUT_SIZE)                   # (1,)

    def forward(self, X):

        self.Z1 = X @ self.W1 + self.b1          # (N, 6)

        self.A1 = np.tanh(self.Z1)               # (N, 6)

        self.Z2 = self.A1 @ self.W2 + self.b2   # (N, 1)
        self.A2 = self.Z2.flatten()              # (N,)
        return self.A2

    def mse_loss(self, X, y):

        y_pred = self.forward(X)
        return float(np.mean((y_pred - y) ** 2))

    def gradients(self, X, y):
  
        N = len(y)
        y_pred = self.forward(X)          

        dL_dA2 = (2 / N) * (y_pred - y)  # (N,)
        dL_dZ2 = dL_dA2.reshape(-1, 1)   # (N, 1)

        dW2 = self.A1.T @ dL_dZ2         # (6, 1)
        db2 = dL_dZ2.sum(axis=0)         # (1,)

        dL_dA1 = dL_dZ2 @ self.W2.T                          # (N, 6)
        dL_dZ1 = dL_dA1 * (1 - np.tanh(self.Z1) ** 2)       # (N, 6)

        dW1 = X.T @ dL_dZ1               # (2, 6)
        db1 = dL_dZ1.sum(axis=0)         # (6,)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def get_flat_weights(self):
        """Flatten all weights into a single 1-D array (length 25)."""
        return np.concatenate([
            self.W1.flatten(),   # 12 values  (indices 0–11)
            self.b1.flatten(),   #  6 values  (indices 12–17)
            self.W2.flatten(),   #  6 values  (indices 18–23)
            self.b2.flatten(),   #  1 value   (index 24)
        ])

    def set_flat_weights(self, w):
        """Restore weights from flat array."""
        self.W1 = w[0:12].reshape(INPUT_SIZE, HIDDEN_SIZE)
        self.b1 = w[12:18]
        self.W2 = w[18:24].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
        self.b2 = w[24:25]

    def get_weight_labels(self):

        labels = []
        for i in range(INPUT_SIZE):
            for j in range(HIDDEN_SIZE):
                labels.append(f"W1[{i},{j}]  (input {FEATURE_NAMES[i]} → hidden {j+1})")
        for j in range(HIDDEN_SIZE):
            labels.append(f"b1[{j}]  (bias of hidden neuron {j+1})")
        for j in range(HIDDEN_SIZE):
            labels.append(f"W2[{j},0]  (hidden {j+1} → output)")
        labels.append("b2[0]  (output bias)")
        return labels

class SGD:

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, net, grads):
        net.W1 -= self.lr * grads["W1"]
        net.b1 -= self.lr * grads["b1"]
        net.W2 -= self.lr * grads["W2"]
        net.b2 -= self.lr * grads["b2"]


class Momentum:

    def __init__(self, lr=0.01, beta=0.9):
        self.lr   = lr
        self.beta = beta
        self.v    = {}   # velocity dictionary, initialized lazily

    def step(self, net, grads):
        for key, g in grads.items():
            if key not in self.v:
                self.v[key] = np.zeros_like(g)
            self.v[key] = self.beta * self.v[key] + self.lr * g
        net.W1 -= self.v["W1"]
        net.b1 -= self.v["b1"]
        net.W2 -= self.v["W2"]
        net.b2 -= self.v["b2"]


class Adagrad:

    def __init__(self, lr=0.01, eps=1e-8):
        self.lr  = lr
        self.eps = eps
        self.G   = {}   # accumulated squared gradients

    def step(self, net, grads):
        for key, g in grads.items():
            if key not in self.G:
                self.G[key] = np.zeros_like(g)
            self.G[key] += g ** 2
        net.W1 -= self.lr * grads["W1"] / (np.sqrt(self.G["W1"]) + self.eps)
        net.b1 -= self.lr * grads["b1"] / (np.sqrt(self.G["b1"]) + self.eps)
        net.W2 -= self.lr * grads["W2"] / (np.sqrt(self.G["W2"]) + self.eps)
        net.b2 -= self.lr * grads["b2"] / (np.sqrt(self.G["b2"]) + self.eps)


class RMSprop:

    def __init__(self, lr=0.001, rho=0.9, eps=1e-8):
        self.lr  = lr
        self.rho = rho
        self.eps = eps
        self.v   = {}

    def step(self, net, grads):
        for key, g in grads.items():
            if key not in self.v:
                self.v[key] = np.zeros_like(g)
            self.v[key] = self.rho * self.v[key] + (1 - self.rho) * g ** 2
        net.W1 -= self.lr * grads["W1"] / (np.sqrt(self.v["W1"]) + self.eps)
        net.b1 -= self.lr * grads["b1"] / (np.sqrt(self.v["b1"]) + self.eps)
        net.W2 -= self.lr * grads["W2"] / (np.sqrt(self.v["W2"]) + self.eps)
        net.b2 -= self.lr * grads["b2"] / (np.sqrt(self.v["b2"]) + self.eps)


class Adam:

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m     = {}   # first moment (mean)
        self.v     = {}   # second moment (variance)
        self.t     = 0    # timestep for bias correction

    def step(self, net, grads):
        self.t += 1
        for key, g in grads.items():
            if key not in self.m:
                self.m[key] = np.zeros_like(g)
                self.v[key] = np.zeros_like(g)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * g ** 2

        # Bias-corrected estimates
        m_hat = {k: self.m[k] / (1 - self.beta1 ** self.t) for k in grads}
        v_hat = {k: self.v[k] / (1 - self.beta2 ** self.t) for k in grads}

        net.W1 -= self.lr * m_hat["W1"] / (np.sqrt(v_hat["W1"]) + self.eps)
        net.b1 -= self.lr * m_hat["b1"] / (np.sqrt(v_hat["b1"]) + self.eps)
        net.W2 -= self.lr * m_hat["W2"] / (np.sqrt(v_hat["W2"]) + self.eps)
        net.b2 -= self.lr * m_hat["b2"] / (np.sqrt(v_hat["b2"]) + self.eps)

OPTIMIZER_CLASSES = {
    "sgd":      lambda lr: SGD(lr=lr),
    "momentum": lambda lr: Momentum(lr=lr),
    "adagrad":  lambda lr: Adagrad(lr=lr),
    "rmsprop":  lambda lr: RMSprop(lr=lr),
    "adam":     lambda lr: Adam(lr=lr),
}

state = {
    "net":           None,   # NeuralNetwork instance
    "optimizer":     None,   # current optimizer instance
    "opt_name":      None,   # string name
    "epoch":         0,
    "loss_history":  [],     # list of float losses per epoch
    "weight_traj":   [],     # list of flat weight arrays (for surface plot)
    "weight_labels": [],     # human-readable weight names
}


def make_fresh_state(opt_name="adam", lr=0.001):
    """Initialize a brand-new network + optimizer."""
    net = NeuralNetwork(seed=42)
    state["net"]           = net
    state["optimizer"]     = OPTIMIZER_CLASSES[opt_name](lr)
    state["opt_name"]      = opt_name
    state["epoch"]         = 0
    state["loss_history"]  = [net.mse_loss(X_DATA, Y_DATA)]
    state["weight_traj"]   = [net.get_flat_weights().tolist()]
    state["weight_labels"] = net.get_weight_labels()

make_fresh_state()

@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/api/info")
def api_info():

    return jsonify({
        "weight_labels": state["weight_labels"],
        "feature_names": FEATURE_NAMES,
        "n_samples":     len(Y_DATA),
        "hidden_size":   HIDDEN_SIZE,
        "input_size":    INPUT_SIZE,
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """
    Reset the network and optimizer.
    Called when user changes optimizer or learning rate.
    """
    data     = request.json
    opt_name = data.get("optimizer", "adam")
    lr       = float(data.get("lr", 0.001))
    make_fresh_state(opt_name, lr)
    return jsonify({"status": "ok", "message": f"Reset with {opt_name}, lr={lr}"})


@app.route("/api/step", methods=["POST"])
def api_step():

    data       = request.json
    n_epochs   = int(data.get("epochs", 1))
    net        = state["net"]
    optimizer  = state["optimizer"]

    for _ in range(n_epochs):
        grads = net.gradients(X_DATA, Y_DATA)   # Step 1: backprop
        optimizer.step(net, grads)               # Step 2: update weights
        loss = net.mse_loss(X_DATA, Y_DATA)      # Step 3a: record loss
        state["loss_history"].append(loss)
        state["weight_traj"].append(            # Step 3b: record weights
            net.get_flat_weights().tolist()
        )
        state["epoch"] += 1

    # Build scatter data: sample 80 points for the prediction plot
    sample_idx  = np.linspace(0, len(X_DATA) - 1, 80, dtype=int)
    X_sample    = X_DATA[sample_idx]
    y_pred_all  = net.forward(X_DATA)

    return jsonify({
        "epoch":        state["epoch"],
        "loss":         round(state["loss_history"][-1], 6),
        "loss_history": [round(v, 6) for v in state["loss_history"]],
        # Scatter: true vs predicted for the sampled points
        "scatter": {
            "y_true": Y_DATA[sample_idx].tolist(),
            "y_pred": y_pred_all[sample_idx].tolist(),
        },
        # Latest weight values (all 25) with their labels
        "weights": {
            "values": net.get_flat_weights().tolist(),
            "labels": state["weight_labels"],
        },
        # Full trajectory for surface plot (every weight snapshot)
        "trajectory": state["weight_traj"],
    })


@app.route("/api/surface", methods=["POST"])
def api_surface():

    data    = request.json
    wi      = int(data.get("weight_i", 0))
    wj      = int(data.get("weight_j", 1))
    grid_n  = 25   # resolution of the surface
    sw      = 1.5  # sweep width ± around current weight value

    net      = state["net"]
    base_w   = net.get_flat_weights()
    tmp      = NeuralNetwork()   # temporary network for surface sweeping

    ci = base_w[wi]   # current value of weight i
    cj = base_w[wj]   # current value of weight j

    wi_vals = np.linspace(ci - sw, ci + sw, grid_n)
    wj_vals = np.linspace(cj - sw, cj + sw, grid_n)

    surface = []
    for vi in wi_vals:
        row = []
        for vj in wj_vals:
            w_copy    = base_w.copy()
            w_copy[wi] = vi
            w_copy[wj] = vj
            tmp.set_flat_weights(w_copy)
            row.append(round(tmp.mse_loss(X_DATA, Y_DATA), 6))
        surface.append(row)

    # Map the trajectory onto the same axes
    traj_i = [w[wi] for w in state["weight_traj"]]
    traj_j = [w[wj] for w in state["weight_traj"]]

    return jsonify({
        "surface":  surface,          # 2D list [grid_n][grid_n]
        "wi_vals":  wi_vals.tolist(),
        "wj_vals":  wj_vals.tolist(),
        "traj_i":   traj_i,
        "traj_j":   traj_j,
        "wi_label": state["weight_labels"][wi],
        "wj_label": state["weight_labels"][wj],
        "current":  {"i": float(ci), "j": float(cj)},
    })
if __name__ == "__main__":
    print("=" * 60)
    print("  Neural Network Optimizer Demo")
    print("  Dataset : California Housing (500 samples)")
    print("  Network : 2 → 6 (tanh) → 1")
    print("  Open    : http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)