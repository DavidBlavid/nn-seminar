"""
For each ABMIL misclassification on the CAMELYON16 test set,
save a two-panel figure: ground truth (left) + attention map (right).

Style mirrors mil-camelyon16.ipynb cell 34.

Output: img/camelyon_misclassified/slide_{idx}_{type}.pdf + .png
"""

import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
CKPT_DIR     = os.path.join(PROJECT_ROOT, 'checkpoints')
DATASET_DIR  = os.path.join(DATA_DIR, 'camelyon16', 'dataset')
OUT_DIR      = os.path.join(PROJECT_ROOT, 'img', 'camelyon_misclassified')
os.makedirs(OUT_DIR, exist_ok=True)


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


# ── load ───────────────────────────────────────────────────────────────────────

print("Loading model …", flush=True)
model = MILModel().to(DEVICE)
model.load_state_dict(
    torch.load(os.path.join(CKPT_DIR, 'camelyon_attn.pt'), weights_only=True))
model.eval()

print("Loading CAMELYON16 test set …", flush=True)
from torchmil.datasets import CAMELYON16MILDataset

test_ds = CAMELYON16MILDataset(
    root=DATASET_DIR, features='resnet50_bt',
    partition='test', bag_keys=['X', 'Y', 'y_inst', 'coords'],
    load_at_init=False,
)
loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                    collate_fn=lambda b: b[0], num_workers=0)

# find optimal threshold from ROC
from sklearn.metrics import roc_curve
probs_all, labels_all = [], []
slides_all = []

print("Running inference …", flush=True)
with torch.no_grad():
    for i, item in enumerate(tqdm(loader, desc='ABMIL', file=_MonitorSink(), mininterval=10.0)):
        X      = item['X'].to(DEVICE)
        y_bag  = int(item['Y'].item())
        logit, attn_w = model(X)
        prob = torch.sigmoid(logit).item()
        probs_all.append(prob)
        labels_all.append(y_bag)
        slides_all.append({
            'idx':    i,
            'prob':   prob,
            'y_bag':  y_bag,
            'coords': item['coords'].numpy(),
            'y_inst': item['y_inst'].numpy().astype(int),
            'attn':   attn_w.cpu().numpy(),
        })

fpr, tpr, thresholds = roc_curve(labels_all, probs_all)
best_thr = thresholds[np.argmax(tpr - fpr)]
print(f"Optimal threshold: {best_thr:.4f}", flush=True)

misclassified = [
    s for s in slides_all
    if int(s['prob'] >= best_thr) != s['y_bag']
]
fp_items = [s for s in misclassified if s['y_bag'] == 0]
fn_items = [s for s in misclassified if s['y_bag'] == 1]
print(f"False positives: {len(fp_items)}  False negatives: {len(fn_items)}", flush=True)


# ── plot ───────────────────────────────────────────────────────────────────────

def save_slide(s, kind):
    coords = s['coords']
    y_inst = s['y_inst']
    attn_w = s['attn']
    idx    = s['idx']
    prob   = s['prob']

    attn_norm = (attn_w - attn_w.min()) / (attn_w.max() - attn_w.min() + 1e-10)
    normal_mask = y_inst == 0
    tumor_mask  = y_inst == 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ground truth
    ax1.scatter(coords[normal_mask, 0], -coords[normal_mask, 1],
                c='steelblue', s=3, alpha=0.5)
    if tumor_mask.any():
        ax1.scatter(coords[tumor_mask, 0], -coords[tumor_mask, 1],
                    c='crimson', s=12, alpha=1.0, zorder=5)
    ax1.set_title(f"Ground truth — crimson = tumor ({tumor_mask.sum()} patches)", fontsize=8)
    ax1.set_aspect('equal'); ax1.axis('off')

    # attention map
    sc = ax2.scatter(coords[:, 0], -coords[:, 1], c=attn_norm,
                     cmap='viridis', vmin=0, vmax=1, s=4, alpha=0.8)
    plt.colorbar(sc, ax=ax2, label='Attention (normalised)', shrink=0.8)
    ax2.set_title('ABMIL attention  (yellow = high)', fontsize=8)
    ax2.set_aspect('equal'); ax2.axis('off')

    true_str = 'pos' if s['y_bag'] == 1 else 'neg'
    pred_str = 'pos' if s['prob'] >= best_thr else 'neg'
    plt.suptitle(
        f"{kind}  |  test slide {idx}  "
        f"(true: {true_str}, pred: {pred_str}, prob: {prob:.3f})",
        fontsize=8
    )
    plt.tight_layout()

    stem = os.path.join(OUT_DIR, f'slide_{idx:03d}_{kind.lower().replace(" ", "_")}')
    plt.savefig(stem + '.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {stem}.png", flush=True)


for s in fp_items:
    save_slide(s, 'false_positive')

for s in fn_items:
    save_slide(s, 'false_negative')

print(f"\nDone. {len(misclassified)} figures in {OUT_DIR}", flush=True)
