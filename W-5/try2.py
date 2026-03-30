import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix


np.random.seed(42)

df = pd.read_csv(r"D:\Deep_Learning\W-5\obesity_data.csv")

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

X     = df[['Age','Gender','Height','Weight','BMI','PhysicalActivityLevel']].values.astype(float)
y_raw = df['ObesityCategory'].values

classes   = ['Normal weight', 'Obese', 'Overweight', 'Underweight']
y_int     = np.array([classes.index(c) for c in y_raw])
n_classes = len(classes)

def to_onehot(y, n):
    oh = np.zeros((len(y), n))
    oh[np.arange(len(y)), y] = 1
    return oh

test_ratio = 0.2
train_idx, test_idx = [], []

for c in range(n_classes):
    c_idx = np.where(y_int == c)[0]
    np.random.shuffle(c_idx)
    n_test = int(len(c_idx) * test_ratio)
    test_idx.extend(c_idx[:n_test])
    train_idx.extend(c_idx[n_test:])

train_idx = np.array(train_idx)
test_idx  = np.array(test_idx)

X_train_raw, X_test_raw = X[train_idx], X[test_idx]
y_train,      y_test     = y_int[train_idx], y_int[test_idx]

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)   
X_test  = scaler.transform(X_test_raw)        


y_train_oh = to_onehot(y_train, n_classes)
y_test_oh  = to_onehot(y_test,  n_classes)

print(f"Train: {X_train.shape}  Test: {X_test.shape}")

def relu(z):      return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def mse_loss(yp, yt): return np.mean((yp - yt)**2)
def mse_grad(yp, yt): return 2*(yp - yt) / yt.shape[0]

def cce_loss(yp, yt):
    return -np.mean(np.sum(yt * np.log(yp + 1e-12), axis=1))

def cce_grad(yp, yt):
    return (yp - yt) / yt.shape[0]

def init_weights(layer_sizes):
    weights, biases = [], []
    for i in range(len(layer_sizes) - 1):
        scale = np.sqrt(2.0 / layer_sizes[i])
        weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale)
        biases.append(np.zeros((1, layer_sizes[i+1])))
    return weights, biases

def forward(X, weights, biases):
    cache_z, cache_a = [], [X]
    a = X
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = a @ W + b
        cache_z.append(z)
        a = relu(z) if i < len(weights) - 1 else softmax(z)
        cache_a.append(a)
    return cache_a[-1], cache_z, cache_a

def backward(y_true, weights, cache_z, cache_a, loss_type):
    y_pred  = cache_a[-1]
    grads_w = [None] * len(weights)
    grads_b = [None] * len(weights)

    if loss_type == 'cce':
        delta = cce_grad(y_pred, y_true)
    else:
        dL = mse_grad(y_pred, y_true)
        s  = y_pred
        delta = s * (dL - (dL * s).sum(axis=1, keepdims=True))

    for i in reversed(range(len(weights))):
        grads_w[i] = cache_a[i].T @ delta
        grads_b[i] = delta.sum(axis=0, keepdims=True)
        if i > 0:
            delta = (delta @ weights[i].T) * relu_grad(cache_z[i-1])

    return grads_w, grads_b

def update(weights, biases, grads_w, grads_b, lr):
    for i in range(len(weights)):
        weights[i] -= lr * grads_w[i]
        biases[i]  -= lr * grads_b[i]
    return weights, biases

def predict(X, weights, biases):
    out, _, _ = forward(X, weights, biases)
    return np.argmax(out, axis=1)

def train_model(layer_sizes, loss_type, batch_size, epochs=200, lr=0.001):
    weights, biases = init_weights(layer_sizes)
    n = X_train.shape[0]
    loss_history = []

    for ep in range(epochs):
        idx  = np.random.permutation(n)
        Xsh, Ysh = X_train[idx], y_train_oh[idx]
        ep_losses = []

        for start in range(0, n, batch_size):
            Xb = Xsh[start:start+batch_size]
            Yb = Ysh[start:start+batch_size]
            pred, cache_z, cache_a = forward(Xb, weights, biases)
            loss = cce_loss(pred, Yb) if loss_type == 'cce' else mse_loss(pred, Yb)
            ep_losses.append(loss)
            gw, gb = backward(Yb, weights, cache_z, cache_a, loss_type)
            weights, biases = update(weights, biases, gw, gb, lr)

        loss_history.append(np.mean(ep_losses))
        if (ep+1) % 20 == 0:
            print(f"  ep {ep+1}/{epochs}  loss={loss_history[-1]:.4f}")

    return weights, biases, loss_history

def compute_metrics(weights, biases, X, y_true):
    y_pred = predict(X, weights, biases)
    acc    = np.mean(y_pred == y_true)
    cm     = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    prec = rec = f1 = 0
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p  = tp/(tp+fp) if (tp+fp) > 0 else 0
        r  = tp/(tp+fn) if (tp+fn) > 0 else 0
        f  = 2*p*r/(p+r) if (p+r) > 0 else 0
        prec += p; rec += r; f1 += f
    return acc, prec/n_classes, rec/n_classes, f1/n_classes, cm

M1 = [6, 64, 32, 16, 4]
M2 = [6, 128, 64, 32, 16, 4]

configs = [
    ('M1','CCE','SGD',  M1,'cce',1),
    ('M1','CCE','Mini', M1,'cce',40),
    ('M1','MSE','SGD',  M1,'mse',1),
    ('M1','MSE','Mini', M1,'mse',40),
    ('M2','CCE','SGD',  M2,'cce',1),
    ('M2','CCE','Mini', M2,'cce',40),
    ('M2','MSE','SGD',  M2,'mse',1),
    ('M2','MSE','Mini', M2,'mse',40),
]

results, histories = [], {}

for (mname, lname, oname, arch, ltype, bs) in configs:
    run_id = f"{mname}-{lname}-{oname}"
    print(f"\n>>> {run_id}")
    W, B, hist = train_model(arch, ltype, bs)
    acc, prec, rec, f1, cm = compute_metrics(W, B, X_test, y_test)
    results.append({'Run': run_id, 'Accuracy': acc,
                    'Macro Precision': prec, 'Macro Recall': rec, 'Macro F1': f1,
                    '_W': W, '_B': B})
    histories[run_id] = hist
    print(f"  Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

results_df = pd.DataFrame(results)

best_idx = results_df['Macro F1'].idxmax()
best_run = results_df.loc[best_idx, 'Run']
best_W   = results_df.loc[best_idx, '_W']
best_B   = results_df.loc[best_idx, '_B']
_, _, _, _, best_cm = compute_metrics(best_W, best_B, X_test, y_test)
print(f"\nBest: {best_run}  F1={results_df.loc[best_idx,'Macro F1']:.4f}")

BG      = '#0a0a0f'
PANEL   = '#0f0f1a'
ACCENT  = '#00f5ff'
GRID    = '#1a1a2e'

PALETTE = [
    '#00f5ff','#ff006e','#fb5607','#8338ec',
    '#3a86ff','#06d6a0','#ffd166','#ef476f'
]
DASHES  = [
    (None,None),(6,2),(3,2,1,2),(2,2),
    (None,None),(6,2),(3,2,1,2),(2,2)
]

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    PANEL,
    'axes.edgecolor':    '#2a2a3e',
    'axes.labelcolor':   '#ccccdd',
    'xtick.color':       '#888899',
    'ytick.color':       '#888899',
    'text.color':        '#e0e0f0',
    'grid.color':        GRID,
    'grid.linewidth':    0.6,
    'legend.facecolor':  '#0f0f1a',
    'legend.edgecolor':  '#2a2a3e',
    'font.family':       'monospace',
})

# ── PLOT 1 : Loss curves ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG)

for i, (rid, hist) in enumerate(histories.items()):
    col  = PALETTE[i]
    dash = DASHES[i]
    lw   = 2.2 if 'M2' in rid else 1.5
    if dash[0] is None:
        ax.plot(hist, color=col, lw=lw, label=rid, alpha=0.9)
    else:
        ax.plot(hist, color=col, lw=lw, label=rid, alpha=0.9, dashes=dash)

ax.set_title('Training Loss — All 8 Runs', fontsize=14, color=ACCENT,
             pad=12, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Loss',  fontsize=11)
ax.grid(True, alpha=0.4)
ax.legend(ncol=2, fontsize=8.5, labelcolor='linecolor')
for spine in ax.spines.values():
    spine.set_edgecolor('#2a2a3e')
plt.tight_layout()
plt.show()

# ── PLOT 2 : Performance table ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.axis('off')

col_labels = ['Run', 'Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']
cell_text  = []
for _, row in results_df.iterrows():
    cell_text.append([
        row['Run'],
        f"{row['Accuracy']:.4f}",
        f"{row['Macro Precision']:.4f}",
        f"{row['Macro Recall']:.4f}",
        f"{row['Macro F1']:.4f}",
    ])

tbl = ax.table(cellText=cell_text, colLabels=col_labels,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 2.0)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#2a2a3e')
    if r == 0:
        cell.set_facecolor('#16213e')
        cell.set_text_props(color=ACCENT, fontweight='bold')
    else:
        is_best = cell_text[r-1][0] == best_run
        cell.set_facecolor('#1a2a1a' if is_best else '#0f0f1a')
        cell.set_text_props(color='#39ff14' if is_best else '#ccccdd')

ax.set_title('Test Set Performance — All 8 Runs', fontsize=13,
             color=ACCENT, pad=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ── PLOT 3 : Confusion matrix ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

im = ax.imshow(best_cm, cmap='plasma', interpolation='nearest')
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(color='#888899')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#888899')

labels = ['Normal\nweight', 'Obese', 'Over\nweight', 'Under\nweight']
ax.set_xticks(range(n_classes));  ax.set_xticklabels(labels, fontsize=9)
ax.set_yticks(range(n_classes));  ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Predicted', fontsize=11, labelpad=8)
ax.set_ylabel('Actual',    fontsize=11, labelpad=8)
ax.set_title(f'Confusion Matrix — Best Model\n{best_run}',
             fontsize=13, color=ACCENT, pad=12, fontweight='bold')

for spine in ax.spines.values():
    spine.set_edgecolor('#2a2a3e')

for i in range(n_classes):
    for j in range(n_classes):
        val = best_cm[i, j]
        txt_col = 'white' if best_cm[i, j] < best_cm.max() * 0.6 else 'black'
        ax.text(j, i, str(val), ha='center', va='center',
                fontsize=12, fontweight='bold', color=txt_col)

plt.tight_layout()
plt.show()

print(f"{'Run':<20} {'Acc':>8} {'Precision':>12} {'Recall':>10} {'F1':>8}")
print("-"*72)
for _, row in results_df.sort_values('Macro F1', ascending=False).iterrows():
    marker = "  <-- BEST" if row['Run'] == best_run else ""
    print(f"{row['Run']:<20} {row['Accuracy']:>8.4f} "
          f"{row['Macro Precision']:>12.4f} {row['Macro Recall']:>10.4f} "
          f"{row['Macro F1']:>8.4f}{marker}")