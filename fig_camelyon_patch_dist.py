"""
Distribution of tumor patch counts across all positive CAMELYON16 test slides.
FN slides (ABMIL) are highlighted.

Output: img/camelyon_patch_dist.png
"""

import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
CKPT_DIR     = os.path.join(PROJECT_ROOT, 'checkpoints')
DATASET_DIR  = os.path.join(DATA_DIR, 'camelyon16', 'dataset')
IMG_DIR      = os.path.join(PROJECT_ROOT, 'img')
os.makedirs(IMG_DIR, exist_ok=True)


class _MonitorSink:
    def write(self, s):
        s = s.strip()
        if s:
            sys.stdout.write(s + '\n')
            sys.stdout.flush()
    def flush(self): pass


# ── model ──────────────────────────────────────────────────────────────────────

class AttentionAggregator(nn.Module):
    def __init__(self, L=512, D=128):
        super().__init__()
        self.V = nn.Linear(L, D); self.U = nn.Linear(L, D); self.w = nn.Linear(D, 1)
    def forward(self, h):
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))
        a = torch.softmax(a, 0)
        return (a * h).sum(0, keepdim=True), a.squeeze(1)

class FeatureProjector(nn.Module):
    def __init__(self, in_dim=2048, out_dim=512, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class MILModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector  = FeatureProjector()
        self.aggregator = AttentionAggregator(512, 128)
        self.classifier = nn.Linear(512, 1)
    def forward(self, X):
        h = self.projector(X)
        z, w = self.aggregator(h)
        return self.classifier(z).squeeze(), w


# ── load & run ─────────────────────────────────────────────────────────────────

print("Loading CAMELYON16 test set …", flush=True)
from torchmil.datasets import CAMELYON16MILDataset

test_ds = CAMELYON16MILDataset(
    root=DATASET_DIR, features='resnet50_bt',
    partition='test', bag_keys=['X', 'Y', 'y_inst'],
    load_at_init=False,
)
loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                    collate_fn=lambda b: b[0], num_workers=0)

print("Loading ABMIL model …", flush=True)
model = MILModel().to(DEVICE)
model.load_state_dict(
    torch.load(os.path.join(CKPT_DIR, 'camelyon_attn.pt'), weights_only=True))
model.eval()

probs_all, labels_all, tumor_counts, bag_sizes = [], [], [], []

print("Running inference …", flush=True)
with torch.no_grad():
    for item in tqdm(loader, desc='ABMIL', file=_MonitorSink(), mininterval=10.0):
        y_bag  = int(item['Y'].item())
        y_inst = item['y_inst'].numpy().astype(int)
        logit, _ = model(item['X'].to(DEVICE))
        prob = torch.sigmoid(logit).item()
        probs_all.append(prob)
        labels_all.append(y_bag)
        tumor_counts.append(int(y_inst.sum()))
        bag_sizes.append(len(y_inst))

fpr, tpr, thresholds = roc_curve(labels_all, probs_all)
best_thr = thresholds[np.argmax(tpr - fpr)]

# per-slide records for positive slides only
pos_records = [
    {'prob': p, 'tumor': t, 'size': s, 'fn': int(p < best_thr)}
    for p, lbl, t, s in zip(probs_all, labels_all, tumor_counts, bag_sizes)
    if lbl == 1
]

fn_counts  = [r['tumor'] for r in pos_records if r['fn']]
tp_counts  = [r['tumor'] for r in pos_records if not r['fn']]

print(f"\nPositive test slides: {len(pos_records)}", flush=True)
print(f"  TP tumor patch counts: {sorted(tp_counts)}", flush=True)
print(f"  FN tumor patch counts: {sorted(fn_counts)}", flush=True)
print(f"  FN max: {max(fn_counts)}  TP min: {min(tp_counts)}", flush=True)


# ── plot ───────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4))

CLIP = 100
tp_counts = [c for c in tp_counts if c <= CLIP]
fn_counts = [c for c in fn_counts if c <= CLIP]
bins = np.arange(0, CLIP + 2, 2)

ax.hist(tp_counts, bins=bins, color='steelblue', alpha=0.85, label=f'True positive  (n={len(tp_counts)})')
ax.hist(fn_counts, bins=bins, color='crimson',   alpha=0.85, label=f'False negative (n={len(fn_counts)})')

ax.set_xlabel('Number of tumor patches per slide', fontsize=10)
ax.set_ylabel('Number of slides', fontsize=10)
ax.set_title('CAMELYON16 test — tumor patch count distribution\n(positive slides only, ABMIL threshold)', fontsize=9)
ax.legend(fontsize=9)

# annotate FN region
if fn_counts:
    ax.axvline(max(fn_counts) + 0.5, color='crimson', linestyle='--',
               linewidth=1, alpha=0.6, label='_')
    ax.text(max(fn_counts) + 1, ax.get_ylim()[1] * 0.95,
            f'all FN ≤ {max(fn_counts)} patches',
            color='crimson', fontsize=8, va='top')

plt.tight_layout()
out = os.path.join(IMG_DIR, 'camelyon_patch_dist.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved → {out}", flush=True)
print("Done.", flush=True)
