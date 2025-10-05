# TCN/ecg_ucr/ecg_ucr_test.py
import argparse, sys, time, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# make top-level TCN modules importable
sys.path.append("../..")

# ---- model: small TCN classifier (TemporalConvNet + GAP + Linear) ----
try:
    from TCN.ecg_ucr.model import TCNClassifier  # if you created model.py as suggested
except Exception:
    from TCN.tcn import TemporalConvNet
    class TCNClassifier(nn.Module):
        def __init__(self, in_channels=1, num_classes=2, levels=4, hidden=64, kernel_size=3, dropout=0.1):
            super().__init__()
            chans = [hidden] * levels
            self.tcn = TemporalConvNet(num_inputs=in_channels, num_channels=chans,
                                       kernel_size=kernel_size, dropout=dropout)
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.fc  = nn.Linear(hidden, num_classes)
        def forward(self, x):  # x: (B,C,T)
            h = self.tcn(x)              # (B,H,T)
            h = self.gap(h).squeeze(-1)  # (B,H)
            return self.fc(h)            # (B,C)

from TCN.common.hartley_tcn import HartleyTCN             # your linear-phase front-end wrapper
from TCN.common.front_end_factory import build_front_end   # now supports multiple FEs

# ---- data loader: generic UCR via sktime (downloads automatically) ----
def load_ucr_numpy3d(name: str, seed=123):
    """
    Returns standardized float32 arrays for any UCR dataset:
      X_train (N,C,T), y_train (N,), X_val, y_val, X_test, y_test
    """
    from sktime.datasets import load_UCR_UEA_dataset  # auto-downloads
    X_tr, y_tr = load_UCR_UEA_dataset(name=name, split="train",
                                      return_X_y=True, return_type="numpy3D")
    X_te, y_te = load_UCR_UEA_dataset(name=name, split="test",
                                      return_X_y=True, return_type="numpy3D")

    # Label-encode to 0..C-1
    classes, y_tr_enc = np.unique(y_tr, return_inverse=True)
    y_te_enc = np.searchsorted(classes, y_te)

    # Stratified 80/20 train/val split
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        (tr_idx, va_idx), = sss.split(X_tr, y_tr_enc)
    except Exception:
        # fallback: random split if sklearn absent
        rng = np.random.RandomState(seed)
        n = X_tr.shape[0]
        idx = rng.permutation(n)
        nv = max(1, int(0.2 * n))
        va_idx, tr_idx = idx[:nv], idx[nv:]

    X_train, y_train = X_tr[tr_idx], y_tr_enc[tr_idx]
    X_val,   y_val   = X_tr[va_idx], y_tr_enc[va_idx]
    X_test,  y_test  = X_te,         y_te_enc

    # z-score using TRAIN stats (per-channel)
    mu = X_train.mean(axis=(0, 2), keepdims=True)
    sd = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    def norm(a): return ((a - mu) / sd).astype(np.float32)

    return norm(X_train), y_train.astype(np.int64), \
           norm(X_val),   y_val.astype(np.int64),   \
           norm(X_test),  y_test.astype(np.int64)

class NumpyTSDataset(torch.utils.data.Dataset):
    def __init__(self, X, y): self.X = torch.from_numpy(X); self.y = torch.from_numpy(y)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

# ---------------- argparse ----------------
p = argparse.ArgumentParser("UCR — TCN classification (bigger dataset ready)")
p.add_argument('--ucr_name', type=str, default='FordA',
               help="UCR dataset name (e.g., FordA, ElectricDevices, Beef, ...)")
p.add_argument('--batch_size', type=int, default=64)
p.add_argument('--epochs', type=int, default=50)
p.add_argument('--seed', type=int, default=123)
p.add_argument('--cuda', action='store_false', help='use CUDA (default: True)')
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--levels', type=int, default=5)
p.add_argument('--hidden', type=int, default=128)
p.add_argument('--ksize', type=int, default=3)
p.add_argument('--dropout', type=float, default=0.1)
p.add_argument('--log_interval', type=int, default=100)

# --- front-end choices ---
p.add_argument('--front_end', type=str, default='none',
    choices=['none', 'lpsconv', 'lpsconv_plus', 'spectral', 'sincnet_bank', 'fir_remez', 'blurpool'],
    help="Prefilter: 'lpsconv' (old), 'lpsconv_plus', 'spectral', 'sincnet_bank', 'fir_remez', 'blurpool', or 'none'.")

# your method (lpsconv) options
p.add_argument('--sym_kernel', type=int, default=21, help='odd kernel length for symmetric FIR')
p.add_argument('--sym_h', type=float, default=1.0)
p.add_argument('--sym_causal', action='store_true')
p.add_argument('--sym_no_residual', action='store_true')

# spectral pooling option
p.add_argument('--spec_cut', type=float, default=0.5, help='keep ratio in rFFT bins (0,1]')

# new front-end knobs
p.add_argument('--fe_k', type=int, default=63, help='generic kernel length for new FEs (odd)')
p.add_argument('--fe_bands', type=int, default=8, help='bands for sincnet_bank')
p.add_argument('--fe_cut', type=float, default=0.5, help='cutoff ratio for fir_remez (0..1)')
p.add_argument('--fe_stride', type=int, default=1, help='stride for blurpool')

args = p.parse_args()

# ---------------- setup ----------------
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

# ---------------- data ----------------
X_tr, y_tr, X_va, y_va, X_te, y_te = load_ucr_numpy3d(name=args.ucr_name, seed=args.seed)
in_channels = X_tr.shape[1]
num_classes = int(max(y_tr.max(), y_va.max(), y_te.max()) + 1)
T = X_tr.shape[2]

train_loader = torch.utils.data.DataLoader(NumpyTSDataset(X_tr, y_tr),
                                           batch_size=args.batch_size, shuffle=True)
val_loader   = torch.utils.data.DataLoader(NumpyTSDataset(X_va, y_va),
                                           batch_size=args.batch_size, shuffle=False)
test_loader  = torch.utils.data.DataLoader(NumpyTSDataset(X_te, y_te),
                                           batch_size=args.batch_size, shuffle=False)

steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)
print(f"[UCR:{args.ucr_name}] train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  "
      f"test={len(test_loader.dataset)}  | C={in_channels} T={T} classes={num_classes}  "
      f"| steps/epoch={steps_per_epoch}")

# ---------------- model ----------------
core = TCNClassifier(in_channels=in_channels, num_classes=num_classes,
                     levels=args.levels, hidden=args.hidden,
                     kernel_size=args.ksize, dropout=args.dropout)

if args.front_end == 'lpsconv':
    from TCN.common.hartley_tcn import HartleyTCN
    model = HartleyTCN(base_tcn=core, in_channels=in_channels,
                       use_front=True, k=args.sym_kernel, h=1.0,
                       causal=args.sym_causal, residual=not args.sym_no_residual)

elif args.front_end in ('lpsconv_plus', 'spectral', 'sincnet_bank', 'fir_remez', 'blurpool'):
    from TCN.common.front_end_factory import build_front_end
    if args.front_end == 'spectral':
        front = build_front_end(kind='spectral', in_channels=in_channels, cutoff_ratio=args.spec_cut)
    elif args.front_end == 'lpsconv_plus':
        front = build_front_end(kind='lpsconv_plus', in_channels=in_channels, k=args.fe_k)
    elif args.front_end == 'sincnet_bank':
        front = build_front_end(kind='sincnet_bank', in_channels=in_channels, k=args.fe_k, bands=args.fe_bands)
    elif args.front_end == 'fir_remez':
        front = build_front_end(kind='fir_remez', in_channels=in_channels, k=args.fe_k, cutoff_ratio=args.fe_cut)
    elif args.front_end == 'blurpool':
        # use a small odd k (e.g., 5) for classic binomial BlurPool; stride controls downsample
        front = build_front_end(kind='blurpool', in_channels=in_channels, k=5, stride=args.fe_stride)
    model = nn.Sequential(front, core)

else:
    model = core

model.to(device)

opt = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    t0 = time.time()
    for bi, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)   # x: (B,C,T)
        if train: opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            opt.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
        if train and bi % args.log_interval == 0:
            ms = (time.time() - t0) * 1000.0 / args.log_interval
            print(f'| batch {bi:5d} | ms/batch {ms:7.2f} | loss {loss.item():.4f}')
            t0 = time.time()
    return total_loss / total, total_correct / total

best_val, best_test = 0.0, 0.0
for epoch in range(1, args.epochs + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader,   train=False)
    te_loss, te_acc = run_epoch(test_loader,  train=False)
    best_val = max(best_val, va_acc); best_test = max(best_test, te_acc)
    print(f'epoch {epoch:02d} | train {tr_loss:.4f}/{tr_acc*100:.2f}% | '
          f'val {va_loss:.4f}/{va_acc*100:.2f}% | test {te_loss:.4f}/{te_acc*100:.2f}% '
          f'| best_val {best_val*100:.2f}% best_test {best_test*100:.2f}%')
