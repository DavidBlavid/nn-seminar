"""
One confusion matrix per (dataset × aggregator) combination — 6 images total.
Threshold chosen to maximise Youden's J on the test ROC curve.

Outputs: img/confusion_{dataset}_{agg}.pdf + .png
"""

import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from sklearn.metrics import roc_curve, confusion_matrix


class _MonitorSink:
    def write(self, s):
        s = s.strip()
        if s:
            sys.stdout.write(s + '\n')
            sys.stdout.flush()
    def flush(self): pass


# ── reproducibility ────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
CKPT_DIR     = os.path.join(PROJECT_ROOT, 'checkpoints')
DATASET_DIR  = os.path.join(DATA_DIR, 'camelyon16', 'dataset')
IMG_DIR      = os.path.join(PROJECT_ROOT, 'img')
os.makedirs(IMG_DIR, exist_ok=True)


# ── shared model classes ───────────────────────────────────────────────────────

class MnistBags(Dataset):
    _MEAN, _STD = 0.1307, 0.3081

    def __init__(self, root, train=True, target_number=9,
                 mean_bag_length=10, var_bag_length=2, num_bags=250, seed=1):
        super().__init__()
        self.target_number = target_number
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self._MEAN,), (self._STD,)),
        ])
        mnist = MNIST(root=root, train=train, transform=transform)
        loader = DataLoader(mnist, batch_size=len(mnist), shuffle=False)
        all_imgs, all_labels = next(iter(loader))
        self._bags, self._bag_labels, self._inst_labels = \
            self._build(all_imgs, all_labels, seed, mean_bag_length, var_bag_length, num_bags)

    def _build(self, imgs, labels, seed, mean_len, var_len, num_bags):
        rng = np.random.default_rng(seed)
        labels_np = labels.numpy()
        non_target = np.where(labels_np != self.target_number)[0]
        bags, bag_lbls, inst_lbls = [], [], []
        next_pos = True
        for _ in range(num_bags):
            length = max(1, int(rng.normal(mean_len, var_len)))
            if next_pos:
                while True:
                    idx = rng.integers(0, len(imgs), size=length)
                    if (labels_np[idx] == self.target_number).any():
                        break
            else:
                idx = rng.choice(non_target, size=length, replace=True)
            il = (labels[idx] == self.target_number)
            bags.append(imgs[idx])
            bag_lbls.append(int(il.any()))
            inst_lbls.append(il)
            next_pos = not next_pos
        return bags, bag_lbls, inst_lbls

    def __len__(self):  return len(self._bags)
    def __getitem__(self, i):
        return self._bags[i], self._bag_labels[i], self._inst_labels[i]


class MnistEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 20, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(nn.Linear(800, 500), nn.ReLU())
    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


class MeanPoolAggregator(nn.Module):
    def forward(self, h): return h.mean(0, keepdim=True), None

class MaxPoolAggregator(nn.Module):
    def forward(self, h): return h.max(0, keepdim=True).values, None

class AttentionAggregator(nn.Module):
    def __init__(self, L=500, D=128):
        super().__init__()
        self.V = nn.Linear(L, D); self.U = nn.Linear(L, D); self.w = nn.Linear(D, 1)
    def forward(self, h):
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))
        a = torch.softmax(a, 0)
        return (a * h).sum(0, keepdim=True), a.squeeze(1)


class MILModelMNIST(nn.Module):
    def __init__(self, aggregator):
        super().__init__()
        self.encoder    = MnistEncoder()
        self.aggregator = aggregator
        self.classifier = nn.Linear(500, 1)
    def forward(self, x):
        h = self.encoder(x)
        z, w = self.aggregator(h)
        return self.classifier(z).squeeze(), w


class FeatureProjector(nn.Module):
    def __init__(self, in_dim=2048, out_dim=512, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class MILModelCAMELYON(nn.Module):
    def __init__(self, aggregator):
        super().__init__()
        self.projector  = FeatureProjector()
        self.aggregator = aggregator
        self.classifier = nn.Linear(512, 1)
    def forward(self, X):
        h = self.projector(X)
        z, w = self.aggregator(h)
        return self.classifier(z).squeeze(), w


# ── load datasets once ─────────────────────────────────────────────────────────

print("Loading MNIST test set …", flush=True)
mnist_test_ds = MnistBags(root=DATA_DIR, train=False, target_number=9,
                           mean_bag_length=10, var_bag_length=2, num_bags=50, seed=3)

print("Loading CAMELYON16 test set …", flush=True)
from torchmil.datasets import CAMELYON16MILDataset
cam_test_ds = CAMELYON16MILDataset(
    root=DATASET_DIR, features='resnet50_bt',
    partition='test', bag_keys=['X', 'Y'],
    load_at_init=False,
)
cam_loader = DataLoader(cam_test_ds, batch_size=1, shuffle=False,
                        collate_fn=lambda b: b[0], num_workers=0)


# ── combinations to run ────────────────────────────────────────────────────────

COMBOS = [
    # (label for title,  agg slug,  checkpoint stem,         model factory)
    ('MNIST',      'Mean',  'mnist_mean',    lambda: MILModelMNIST(MeanPoolAggregator())),
    ('MNIST',      'Max',   'mnist_max',     lambda: MILModelMNIST(MaxPoolAggregator())),
    ('MNIST',      'ABMIL', 'mnist_attn',    lambda: MILModelMNIST(AttentionAggregator(500, 128))),
    ('CAMELYON16', 'Mean',  'camelyon_mean', lambda: MILModelCAMELYON(MeanPoolAggregator())),
    ('CAMELYON16', 'Max',   'camelyon_max',  lambda: MILModelCAMELYON(MaxPoolAggregator())),
    ('CAMELYON16', 'ABMIL', 'camelyon_attn', lambda: MILModelCAMELYON(AttentionAggregator(512, 128))),
]


# ── helper: run inference ──────────────────────────────────────────────────────

def run_mnist(model):
    probs, labels = [], []
    with torch.no_grad():
        for bag, label, _ in mnist_test_ds:
            logit, _ = model(bag.to(DEVICE))
            probs.append(torch.sigmoid(logit).item())
            labels.append(label)
    return probs, labels

def run_camelyon(model, desc):
    probs, labels = [], []
    with torch.no_grad():
        for item in tqdm(cam_loader, desc=desc, file=_MonitorSink(), mininterval=10.0):
            logit, _ = model(item['X'].to(DEVICE))
            probs.append(torch.sigmoid(logit).item())
            labels.append(int(item['Y'].item()))
    return probs, labels


# ── plot helper ────────────────────────────────────────────────────────────────

def save_confusion(cm, title, out_stem):
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    im = ax.imshow(cm, cmap='Blues', vmin=0)
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() * 0.6 else '#222'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=18, fontweight='bold', color=color)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Negative', 'Positive'])
    ax.set_xlabel('Predicted label', fontsize=9)
    ax.set_ylabel('True label', fontsize=9)
    ax.set_title(title, fontsize=9, pad=6)
    plt.tight_layout()
    plt.savefig(out_stem + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


# ── main loop ──────────────────────────────────────────────────────────────────

best_cms = {}   # keyed by (dataset, agg_name) → (cm, best_thr)

for dataset, agg_name, ckpt_stem, make_model in COMBOS:
    tag = f"{dataset.lower().replace('camelyon16', 'cam')}_{agg_name.lower()}"
    print(f"\n[{tag}] loading checkpoint …", flush=True)

    model = make_model().to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(CKPT_DIR, f'{ckpt_stem}.pt'), weights_only=True))
    model.eval()

    if dataset == 'MNIST':
        probs, labels = run_mnist(model)
    else:
        probs, labels = run_camelyon(model, desc=tag)

    fpr, tpr, thresholds = roc_curve(labels, probs)
    best_thr = thresholds[np.argmax(tpr - fpr)]
    preds = [int(p >= best_thr) for p in probs]
    cm = confusion_matrix(labels, preds)

    tn, fp, fn, tp = cm.ravel()
    print(f"[{tag}]  thr={best_thr:.4f}  TN={tn} FP={fp} FN={fn} TP={tp}", flush=True)

    title = f'{agg_name}  –  {dataset}\n(threshold = {best_thr:.3f})'
    out_stem = os.path.join(IMG_DIR, f'confusion_{tag}')
    save_confusion(cm, title, out_stem)
    print(f"[{tag}]  saved → {out_stem}.png", flush=True)

    best_cms[(dataset, agg_name)] = (cm, best_thr)


# ── combined figure: best model per dataset (ABMIL/MNIST + Max/CAMELYON16) ────

print("\nSaving combined confusion_matrices …", flush=True)
pairs = [
    ('MNIST',      'ABMIL'),
    ('CAMELYON16', 'Max'),
]
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
for ax, (dataset, agg_name) in zip(axes, pairs):
    cm, best_thr = best_cms[(dataset, agg_name)]
    ax.imshow(cm, cmap='Blues', vmin=0)
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() * 0.6 else '#222'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=16, fontweight='bold', color=color)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Negative', 'Positive'])
    ax.set_xlabel('Predicted label', fontsize=9)
    ax.set_ylabel('True label', fontsize=9)
    ax.set_title(f'{agg_name}  –  {dataset}\n(threshold = {best_thr:.3f})', fontsize=9, pad=6)

plt.tight_layout()
out_stem = os.path.join(IMG_DIR, 'confusion_matrices')
plt.savefig(out_stem + '.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved → {out_stem}.png", flush=True)

print("\nDone.", flush=True)
