import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
df = pd.read_csv(r"D:\Deep_Learning\W-5\obesity_data.csv")

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

X = df[['Age','Gender','Height','Weight','BMI','PhysicalActivityLevel']].values.astype(float)
y_raw = df['ObesityCategory'].values

classes = ['Normal weight','Obese','Overweight','Underweight']
y_int = np.array([classes.index(c) for c in y_raw])
n_classes = len(classes)

def to_onehot(y, n):
    oh = np.zeros((len(y), n))
    oh[np.arange(len(y)), y] = 1
    return oh

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_int, test_size=0.2, random_state=42, stratify=y_int)

y_train_oh = to_onehot(y_train, n_classes)
y_test_oh  = to_onehot(y_test,  n_classes)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Classes: {classes}")

# ─────────────────────────────────────────────
# 2. ACTIVATION FUNCTIONS
# ─────────────────────────────────────────────
def relu(z):       return np.maximum(0, z)
def relu_grad(z):  return (z > 0).astype(float)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

# ─────────────────────────────────────────────
# 3. LOSS FUNCTIONS
# ─────────────────────────────────────────────
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def mse_grad(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def cce_loss(y_pred, y_true):
    eps = 1e-12
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

def cce_grad(y_pred, y_true):
    # combined softmax + CCE gradient = (y_pred - y_true) / batch_size
    return (y_pred - y_true) / y_true.shape[0]

# ─────────────────────────────────────────────
# 4. NEURAL NETWORK CLASS
# ─────────────────────────────────────────────
class FeedforwardNN:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes)-1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i+1]
            # He init for ReLU layers
            scale = np.sqrt(2.0 / fan_in)
            W = np.random.randn(fan_in, fan_out) * scale
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        self.cache_z = []
        self.cache_a = [X]
        a = X
        # all hidden layers use ReLU; last layer uses softmax
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            self.cache_z.append(z)
            if i < len(self.weights) - 1:
                a = relu(z)
            else:
                a = softmax(z)
            self.cache_a.append(a)
        return a

    def backward(self, y_true, loss_type='cce'):
        y_pred = self.cache_a[-1]
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        if loss_type == 'cce':
            # softmax + CCE combined gradient
            delta = cce_grad(y_pred, y_true)
        else:
            # MSE gradient through softmax
            dL_da = mse_grad(y_pred, y_true)
            # softmax jacobian: diag(s) - s*s^T  simplified:
            s = y_pred
            delta = s * (dL_da - (dL_da * s).sum(axis=1, keepdims=True))

        for i in reversed(range(len(self.weights))):
            a_prev = self.cache_a[i]
            grads_w[i] = a_prev.T @ delta
            grads_b[i] = delta.sum(axis=0, keepdims=True)
            if i > 0:
                dz = delta @ self.weights[i].T
                delta = dz * relu_grad(self.cache_z[i-1])

        return grads_w, grads_b

    def update(self, grads_w, grads_b, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads_w[i]
            self.biases[i]  -= lr * grads_b[i]

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# ─────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────
def train(layer_sizes, loss_type, batch_size, epochs=120, lr=0.005):
    model = FeedforwardNN(layer_sizes)
    n = X_train.shape[0]
    loss_history = []

    for ep in range(epochs):
        idx = np.random.permutation(n)
        X_sh, y_sh = X_train[idx], y_train_oh[idx]
        ep_losses = []
        for start in range(0, n, batch_size):
            Xb = X_sh[start:start+batch_size]
            yb = y_sh[start:start+batch_size]
            pred = model.forward(Xb)
            if loss_type == 'cce':
                loss = cce_loss(pred, yb)
            else:
                loss = mse_loss(pred, yb)
            ep_losses.append(loss)
            gw, gb = model.backward(yb, loss_type)
            model.update(gw, gb, lr)
        loss_history.append(np.mean(ep_losses))

        if (ep+1) % 20 == 0:
            print(f"  ep {ep+1}/{epochs}  loss={loss_history[-1]:.4f}")

    return model, loss_history

# ─────────────────────────────────────────────
# 6. METRICS
# ─────────────────────────────────────────────
def compute_metrics(model, X, y_true):
    y_pred = model.predict(X)
    acc = np.mean(y_pred == y_true)
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    prec = rec = f1 = 0
    for c in range(n_classes):
        tp = cm[c,c]
        fp = cm[:,c].sum() - tp
        fn = cm[c,:].sum() - tp
        p = tp/(tp+fp) if (tp+fp) > 0 else 0
        r = tp/(tp+fn) if (tp+fn) > 0 else 0
        f = 2*p*r/(p+r) if (p+r) > 0 else 0
        prec += p; rec += r; f1 += f
    return acc, prec/n_classes, rec/n_classes, f1/n_classes, cm

# ─────────────────────────────────────────────
# 7. RUN ALL 8 EXPERIMENTS
# ─────────────────────────────────────────────
M1 = [6, 64, 32, 16, 4]
M2 = [6, 128, 64, 32, 16, 4]

configs = [
    ('M1','CCE','SGD',   M1,'cce',1),
    ('M1','CCE','Mini',  M1,'cce',40),
    ('M1','MSE','SGD',   M1,'mse',1),
    ('M1','MSE','Mini',  M1,'mse',40),
    ('M2','CCE','SGD',   M2,'cce',1),
    ('M2','CCE','Mini',  M2,'cce',40),
    ('M2','MSE','SGD',   M2,'mse',1),
    ('M2','MSE','Mini',  M2,'mse',40),
]

results   = []
histories = {}

for (mname, lname, oname, arch, ltype, bs) in configs:
    run_id = f"{mname}-{lname}-{oname}"
    print(f"\n>>> {run_id}")
    model, hist = train(arch, ltype, bs)
    acc, prec, rec, f1, cm = compute_metrics(model, X_test, y_test)
    results.append({
        'Run': run_id, 'Model': mname, 'Loss': lname, 'Optimizer': oname,
        'Accuracy': acc, 'Macro Precision': prec,
        'Macro Recall': rec, 'Macro F1': f1
    })
    histories[run_id] = hist
    print(f"  Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

results_df = pd.DataFrame(results)

# ─────────────────────────────────────────────
# 8. FIND BEST MODEL
# ─────────────────────────────────────────────
best_idx  = results_df['Macro F1'].idxmax()
best_run  = results_df.loc[best_idx, 'Run']
best_row  = results_df.loc[best_idx]
print(f"\nBest run: {best_run}  F1={best_row['Macro F1']:.4f}")

# Re-run best model to get its confusion matrix
best_cfg = next(c for c in configs if f"{c[0]}-{c[1]}-{c[2]}" == best_run)
best_model, _ = train(best_cfg[3], best_cfg[4], best_cfg[5])
_, _, _, _, best_cm = compute_metrics(best_model, X_test, y_test)

# ─────────────────────────────────────────────
# 9. PLOTS  (shown as dialog boxes, not saved)
# ─────────────────────────────────────────────

# --- Plot 1: Combined loss curves ---
plt.figure(figsize=(10, 5))
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
for i, (run_id, hist) in enumerate(histories.items()):
    plt.plot(hist, linestyle=linestyles[i], label=run_id)
plt.title('Training Loss - All 8 Runs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

# --- Plot 2: Performance table ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
col_labels = ['Run', 'Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']
cell_text = []
for _, row in results_df.iterrows():
    cell_text.append([
        row['Run'],
        f"{row['Accuracy']:.4f}",
        f"{row['Macro Precision']:.4f}",
        f"{row['Macro Recall']:.4f}",
        f"{row['Macro F1']:.4f}",
    ])
table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)
ax.set_title('Test Set Performance - All 8 Runs', pad=12)
plt.tight_layout()
plt.show()

# --- Plot 3: Confusion matrix for best model ---
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(best_cm, cmap='Blues')
plt.colorbar(im, ax=ax)
short = ['Normal weight', 'Obese', 'Overweight', 'Underweight']
ax.set_xticks(range(n_classes))
ax.set_xticklabels(short, rotation=15, fontsize=8)
ax.set_yticks(range(n_classes))
ax.set_yticklabels(short, fontsize=8)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix - Best Model ({best_run})')
for i in range(n_classes):
    for j in range(n_classes):
        ax.text(j, i, str(best_cm[i, j]), ha='center', va='center', fontsize=10)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# 10. PRINT RESULTS TABLE
# ─────────────────────────────────────────────
print("\n" + "="*72)
print(f"{'Run':<20} {'Acc':>8} {'Precision':>12} {'Recall':>10} {'F1':>8}")
print("-"*72)
for _, row in results_df.sort_values('Macro F1', ascending=False).iterrows():
    marker = " *best*" if row['Run'] == best_run else ""
    print(f"{row['Run']:<20} {row['Accuracy']:>8.4f} "
          f"{row['Macro Precision']:>12.4f} {row['Macro Recall']:>10.4f} "
          f"{row['Macro F1']:>8.4f}{marker}")
print("="*72)