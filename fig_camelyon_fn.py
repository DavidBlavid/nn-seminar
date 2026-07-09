"""
False-negative analysis for CAMELYON16.

Figure 1 — img/camelyon_roc.pdf
    Bag-level ROC curves for all three aggregators.
    The operating point (optimal Youden threshold) is marked, with FN slides
    annotated by their predicted probability.

Figure 2 — img/camelyon_fn_instance_roc.pdf
    For ABMIL only: instance-level ROC curves (attention weight vs patch label)
    for each false-negative slide. Shows whether attention still localises
    tumour tissue on slides the bag-level classifier missed.
"""

import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score

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


# ── model classes ──────────────────────────────────────────────────────────────

class MeanPoolAggregator(nn.Module):
    def forward(self, h): return h.mean(0, keepdim=True), None

class MaxPoolAggregator(nn.Module):
    def forward(self, h): return h.max(0, keepdim=True).values, None

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
    def __init__(self, aggregator):
        super().__init__()
        self.projector  = FeatureProjector()
        self.aggregator = aggregator
        self.classifier = nn.Linear(512, 1)
    def forward(self, X):
        h = self.projector(X)
        z, w = self.aggregator(h)
        return self.classifier(z).squeeze(), w


# ── load dataset (with instance labels) ───────────────────────────────────────

print("Loading CAMELYON16 test set …", flush=True)
from torchmil.datasets import CAMELYON16MILDataset

cam_test_ds = CAMELYON16MILDataset(
    root=DATASET_DIR, features='resnet50_bt',
    partition='test', bag_keys=['X', 'Y', 'y_inst'],
    load_at_init=False,
)
cam_loader = DataLoader(cam_test_ds, batch_size=1, shuffle=False,
                        collate_fn=lambda b: b[0], num_workers=0)

COMBOS = [
    ('Mean',  'camelyon_mean', MeanPoolAggregator()),
    ('Max',   'camelyon_max',  MaxPoolAggregator()),
    ('ABMIL', 'camelyon_attn', AttentionAggregator(512, 128)),
]
COLORS = {'Mean': '#4c72b0', 'Max': '#dd8452', 'ABMIL': '#55a868'}


# ── inference for all models ───────────────────────────────────────────────────

results = {}   # name → {probs, labels, weights_pos, y_inst_pos}

for name, ckpt, agg in COMBOS:
    print(f"\nRunning {name} …", flush=True)
    model = MILModel(agg).to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(CKPT_DIR, f'{ckpt}.pt'), weights_only=True))
    model.eval()

    probs, labels, slide_data = [], [], []
    with torch.no_grad():
        for item in tqdm(cam_loader, desc=name, file=_MonitorSink(), mininterval=10.0):
            X      = item['X'].to(DEVICE)
            y_bag  = int(item['Y'].item())
            y_inst = item['y_inst'].numpy().astype(int)
            logit, attn = model(X)
            prob = torch.sigmoid(logit).item()
            w    = attn.cpu().numpy() if attn is not None else None
            probs.append(prob)
            labels.append(y_bag)
            slide_data.append({'prob': prob, 'y_bag': y_bag, 'y_inst': y_inst, 'w': w})

    results[name] = {'probs': probs, 'labels': labels, 'slides': slide_data}
    auc = roc_auc_score(labels, probs)
    print(f"  Bag AUC = {auc:.4f}", flush=True)


# ── Figure 1: bag-level ROC curves with FN operating points ───────────────────

print("\nPlotting bag-level ROC …", flush=True)
fig, ax = plt.subplots(figsize=(5, 5))

for name, res in results.items():
    probs  = res['probs']
    labels = res['labels']
    color  = COLORS[name]

    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc  = roc_auc_score(labels, probs)
    best = np.argmax(tpr - fpr)
    best_thr = thresholds[best]

    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f'{name}  (AUC = {auc:.4f})')
    ax.scatter(fpr[best], tpr[best], color=color, s=80, zorder=5)

    # annotate each FN slide near the operating point
    slides = res['slides']
    fn_probs = [s['prob'] for s in slides if s['y_bag'] == 1 and s['prob'] < best_thr]
    print(f"  {name}: {len(fn_probs)} FN  probs={[f'{p:.3f}' for p in fn_probs]}", flush=True)

ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title('CAMELYON16 — Bag-level ROC', fontsize=10)
ax.legend(fontsize=9, loc='lower right')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
out = os.path.join(IMG_DIR, 'camelyon_roc')
plt.savefig(out + '.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved → {out}.png", flush=True)


# ── Figure 2: ABMIL instance-level ROC for FN slides ──────────────────────────

print("\nPlotting ABMIL instance-level ROC for FN slides …", flush=True)

abmil_slides = results['ABMIL']['slides']
abmil_probs  = results['ABMIL']['probs']
abmil_labels = results['ABMIL']['labels']

fpr_bag, tpr_bag, thr_bag = roc_curve(abmil_labels, abmil_probs)
best_thr_abmil = thr_bag[np.argmax(tpr_bag - fpr_bag)]

fn_slides = [s for s in abmil_slides
             if s['y_bag'] == 1 and s['prob'] < best_thr_abmil
             and s['w'] is not None
             and 0 < s['y_inst'].sum() < len(s['y_inst'])]

print(f"  ABMIL FN slides with usable instance labels: {len(fn_slides)}", flush=True)

fig, ax = plt.subplots(figsize=(5, 5))

all_fpr_interp = np.linspace(0, 1, 200)
tpr_curves = []

for s in fn_slides:
    fpr_i, tpr_i, _ = roc_curve(s['y_inst'], s['w'])
    auc_i = roc_auc_score(s['y_inst'], s['w'])
    ax.plot(fpr_i, tpr_i, alpha=0.55, linewidth=1.2,
            label=f'prob={s["prob"]:.3f}  inst-AUC={auc_i:.3f}')
    tpr_interp = np.interp(all_fpr_interp, fpr_i, tpr_i)
    tpr_curves.append(tpr_interp)

if tpr_curves:
    mean_tpr = np.mean(tpr_curves, axis=0)
    mean_auc = roc_auc_score(
        np.concatenate([s['y_inst'] for s in fn_slides]),
        np.concatenate([s['w']      for s in fn_slides])
    )
    ax.plot(all_fpr_interp, mean_tpr, 'k-', linewidth=2.5,
            label=f'Mean  (inst-AUC = {mean_auc:.3f})')
    print(f"  Mean instance AUC (FN slides): {mean_auc:.4f}", flush=True)

ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title(f'ABMIL — Instance-level ROC on false-negative slides\n'
             f'(n={len(fn_slides)}, attention weight vs patch label)',
             fontsize=9)
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
out = os.path.join(IMG_DIR, 'camelyon_fn_instance_roc')
plt.savefig(out + '.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved → {out}.png", flush=True)

print("\nDone.", flush=True)
