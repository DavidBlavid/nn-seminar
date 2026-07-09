"""
Two MNIST bag visualisations for the paper, styled to match the notebook:
  1. Positive bag with the most 9s  (sorted by attention, highest first)
  2. Best misclassified bag          (false negative preferred)

Border colour = RdYlGn(attention_normalised).
Score shown as title above each instance (red = digit 9, black = other).

Output: img/mnist_multi9.pdf + img/mnist_multi9.png
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

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
CKPT_DIR     = os.path.join(PROJECT_ROOT, 'checkpoints')
IMG_DIR      = os.path.join(PROJECT_ROOT, 'img')
os.makedirs(IMG_DIR, exist_ok=True)

CMAP = plt.cm.RdYlGn


class _MonitorSink:
    def write(self, s):
        s = s.strip()
        if s:
            sys.stdout.write(s + '\n')
            sys.stdout.flush()
    def flush(self): pass


# ── model / dataset ────────────────────────────────────────────────────────────

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
        self._bags, self._bag_labels, self._inst_labels, self._digit_labels = \
            self._build(all_imgs, all_labels, seed, mean_bag_length, var_bag_length, num_bags)

    def _build(self, imgs, labels, seed, mean_len, var_len, num_bags):
        rng = np.random.default_rng(seed)
        labels_np = labels.numpy()
        non_target = np.where(labels_np != self.target_number)[0]
        bags, bag_lbls, inst_lbls, digit_lbls = [], [], [], []
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
            digit_lbls.append(labels[idx])
            next_pos = not next_pos
        return bags, bag_lbls, inst_lbls, digit_lbls

    def __len__(self):  return len(self._bags)
    def __getitem__(self, i):
        return (self._bags[i], self._bag_labels[i],
                self._inst_labels[i], self._digit_labels[i])


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


class AttentionAggregator(nn.Module):
    def __init__(self, L=500, D=128):
        super().__init__()
        self.V = nn.Linear(L, D); self.U = nn.Linear(L, D); self.w = nn.Linear(D, 1)
    def forward(self, h):
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))
        a = torch.softmax(a, 0)
        return (a * h).sum(0, keepdim=True), a.squeeze(1)


class MILModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = MnistEncoder()
        self.aggregator = AttentionAggregator()
        self.classifier = nn.Linear(500, 1)
    def forward(self, x):
        h = self.encoder(x)
        z, w = self.aggregator(h)
        return self.classifier(z).squeeze(), w


_MEAN_T = torch.tensor([0.1307])
_STD_T  = torch.tensor([0.3081])
def denorm(t): return (t * _STD_T + _MEAN_T).clamp(0, 1)


# ── load ───────────────────────────────────────────────────────────────────────

print("Loading model …", flush=True)
model = MILModel().to(DEVICE)
model.load_state_dict(
    torch.load(os.path.join(CKPT_DIR, 'mnist_attn.pt'), weights_only=True))
model.eval()

print("Loading test set …", flush=True)
test_ds = MnistBags(root=DATA_DIR, train=False, target_number=9,
                    mean_bag_length=10, var_bag_length=2, num_bags=50, seed=3)

print("Running inference …", flush=True)
records = []
with torch.no_grad():
    for i, (bag, label, il, dl) in enumerate(
            tqdm(test_ds, desc='bags', file=_MonitorSink(), mininterval=2.0)):
        logit, attn = model(bag.to(DEVICE))
        prob    = torch.sigmoid(logit).item()
        pred    = int(prob >= 0.5)
        records.append((i, label, pred, prob, attn.cpu().numpy(), bag, il, dl))


# ── select bags ────────────────────────────────────────────────────────────────

multi9 = sorted([r for r in records if r[1] == 1], key=lambda r: -r[6].sum())
bag_multi9 = multi9[0]
print(f"Multi-9 bag  index={bag_multi9[0]}  n_nines={int(bag_multi9[6].sum())}  "
      f"prob={bag_multi9[3]:.3f}", flush=True)

fn = [r for r in records if r[1] == 1 and r[2] == 0]
fp = [r for r in records if r[1] == 0 and r[2] == 1]
if not fn and not fp:
    raise RuntimeError("No misclassified bags found.")
bag_misc   = min(fn, key=lambda r: r[3]) if fn else max(fp, key=lambda r: r[3])
misc_type  = "False Negative" if fn else "False Positive"
print(f"Misclassified  type={misc_type}  index={bag_misc[0]}  "
      f"prob={bag_misc[3]:.3f}", flush=True)


# ── plot helper ────────────────────────────────────────────────────────────────

def plot_bag(bag_record, suptitle, out_stem):
    i, label, pred, prob, attn, bag, il, dl = bag_record
    order       = np.argsort(attn)[::-1].copy()
    attn_sorted = attn[order]
    imgs_sorted = bag[order]
    is9_sorted  = il.numpy()[order]
    attn_norm   = (attn_sorted - attn_sorted.min()) / \
                  (attn_sorted.max() - attn_sorted.min() + 1e-8)

    n = len(order)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.4, 1.9))
    if n == 1:
        axes = [axes]

    for j, (ax, img, is9) in enumerate(zip(axes, imgs_sorted, is9_sorted)):
        ax.imshow(denorm(img).squeeze(), cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(CMAP(attn_norm[j]))
            spine.set_linewidth(4)
        ax.set_title(f'{attn_sorted[j]:.3f}', fontsize=7,
                     color='red' if is9 else 'black', pad=2)

    plt.suptitle(suptitle, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_stem + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_stem}.png", flush=True)


# ── generate figures ───────────────────────────────────────────────────────────

n9 = int(bag_multi9[6].sum())
plot_bag(bag_multi9,
         suptitle=f'Bag {bag_multi9[0]} — sorted by attention  ({n9} × digit 9)',
         out_stem=os.path.join(IMG_DIR, 'mnist_multi9'))

true_str = 'pos' if bag_misc[1] == 1 else 'neg'
plot_bag(bag_misc,
         suptitle=f'Bag {bag_misc[0]} — {misc_type}  (true: {true_str}, prob: {bag_misc[3]:.3f})',
         out_stem=os.path.join(IMG_DIR, 'mnist_misclassified'))

print("Done.", flush=True)
