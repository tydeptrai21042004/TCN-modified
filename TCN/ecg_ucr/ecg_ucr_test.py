# TCN/ecg_ucr/ecg_ucr_test.py
import argparse, sys, time, math, warnings, os
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

# ---------------- dataset loaders ----------------
def _zscore_train_stats(Xtr):
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
    return mu, sd

def _apply_zscore(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)

def load_ucr_numpy3d(name: str, seed=123):
    """Any UCR/UEA dataset via sktime, unified to (N,C,T)."""
    from sktime.datasets import load_UCR_UEA_dataset
    X_tr, y_tr = load_UCR_UEA_dataset(name=name, split="train",
                                      return_X_y=True, return_type="numpy3D")
    X_te, y_te = load_UCR_UEA_dataset(name=name, split="test",
                                      return_X_y=True, return_type="numpy3D")

    classes, y_tr_enc = np.unique(y_tr, return_inverse=True)
    y_te_enc = np.searchsorted(classes, y_te)

    # Stratified 80/20 train/val split
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        (tr_idx, va_idx), = sss.split(X_tr, y_tr_enc)
    except Exception:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(X_tr.shape[0]); nv = max(1, int(0.2 * X_tr.shape[0]))
        va_idx, tr_idx = idx[:nv], idx[nv:]

    X_train, y_train = X_tr[tr_idx], y_tr_enc[tr_idx]
    X_val,   y_val   = X_tr[va_idx], y_tr_enc[va_idx]
    X_test,  y_test  = X_te,         y_te_enc

    mu, sd = _zscore_train_stats(X_train)
    return _apply_zscore(X_train, mu, sd), y_train.astype(np.int64), \
           _apply_zscore(X_val,   mu, sd), y_val.astype(np.int64),   \
           _apply_zscore(X_test,  mu, sd), y_test.astype(np.int64)

def load_speechcommands(root: str, sample_rate=16000, seconds=1.0):
    """Google SpeechCommands v2 via torchaudio; fixed-length mono waveforms -> (N,1,T)."""
    import torchaudio
    from torchaudio.datasets import SPEECHCOMMANDS

    # official split files exist inside dataset; helper to map split name
    class SubsetSC(SPEECHCOMMANDS):
        def __init__(self, subset: str, **kw):
            super().__init__(**kw)
            def load_list(filename):
                filepath = os.path.join(self._path, filename)
                with open(filepath) as f: return [os.path.join(self._path, l.strip()) for l in f]
            if subset == "validation":
                self._walker = load_list("validation_list.txt")
            elif subset == "testing":
                self._walker = load_list("testing_list.txt")
            elif subset == "training":
                excludes = set(load_list("validation_list.txt") + load_list("testing_list.txt"))
                self._walker = [w for w in self._walker if os.path.join(self._path, w) not in excludes]
            else:
                raise ValueError("subset must be 'training'/'validation'/'testing'")

    target_len = int(sample_rate * seconds)
    resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=sample_rate)

    def _to_numpy(ds):
        xs, ys, labels = [], [], sorted(list(set(i[2] for i in ds)))  # label strings
        lab2id = {lab:i for i,lab in enumerate(labels)}
        for wav, sr, label, *_ in ds:
            if sr != sample_rate: wav = resample(wav)
            wav = wav.mean(dim=0, keepdim=True)          # (1, T)
            T = wav.shape[-1]
            if T < target_len:
                wav = torch.nn.functional.pad(wav, (0, target_len - T))
            else:
                wav = wav[..., :target_len]
            xs.append(wav.numpy()); ys.append(lab2id[label])
        X = np.stack(xs, axis=0).astype(np.float32)      # (N,1,T)
        y = np.array(ys, dtype=np.int64)
        return X, y, labels

    train = SubsetSC(subset="training",  root=root, download=True)
    valid = SubsetSC(subset="validation",root=root, download=True)
    test  = SubsetSC(subset="testing",   root=root, download=True)

    Xtr, ytr, labels = _to_numpy(train)
    Xva, yva, _      = _to_numpy(valid)
    Xte, yte, _      = _to_numpy(test)

    # z-score using train stats
    mu, sd = _zscore_train_stats(Xtr)
    return _apply_zscore(Xtr, mu, sd), ytr, \
           _apply_zscore(Xva, mu, sd), yva, \
           _apply_zscore(Xte, mu, sd), yte, labels

def load_gtzan(root: str, sample_rate=16000, seconds=5.0):
    """GTZAN (10 genres) via torchaudio; fixed-length mono waveforms -> (N,1,T)."""
    import torchaudio
    from torchaudio.datasets import GTZAN
    target_len = int(sample_rate * seconds)
    resample = torchaudio.transforms.Resample(orig_freq=22050, new_freq=sample_rate)

    def _to_numpy(ds):
        xs, ys, labels = [], [], sorted(list(set(d[2] for d in ds)))
        lab2id = {lab:i for i,lab in enumerate(labels)}
        for waveform, sr, label in ds:
            if sr != sample_rate: waveform = resample(waveform)
            wav = waveform.mean(dim=0, keepdim=True)   # mono
            T = wav.shape[-1]
            if T < target_len:
                wav = torch.nn.functional.pad(wav, (0, target_len - T))
            else:
                wav = wav[..., :target_len]
            xs.append(wav.numpy()); ys.append(lab2id[label])
        X = np.stack(xs, axis=0).astype(np.float32); y = np.array(ys, dtype=np.int64)
        return X, y, labels

    train = GTZAN(root=root, subset='train', download=True)
    test  = GTZAN(root=root, subset='test',  download=True)
    # no official val split; use 20% of train as val (stratified)
    Xtr_all, ytr_all, labels = _to_numpy(train)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
    (tr_idx, va_idx), = sss.split(Xtr_all, ytr_all)
    Xtr, ytr = Xtr_all[tr_idx], ytr_all[tr_idx]
    Xva, yva = Xtr_all[va_idx], ytr_all[va_idx]
    Xte, yte, _ = _to_numpy(test)

    mu, sd = _zscore_train_stats(Xtr)
    return _apply_zscore(Xtr, mu, sd), ytr, \
           _apply_zscore(Xva, mu, sd), yva, \
           _apply_zscore(Xte, mu, sd), yte, labels

class NumpyTSDataset(torch.utils.data.Dataset):
    def __init__(self, X, y): self.X = torch.from_numpy(X); self.y = torch.from_numpy(y)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

# ---------------- argparse ----------------
p = argparse.ArgumentParser("TCN classification across multiple time-series datasets")
p.add_argument('--source', type=str, default='ucr',
               choices=['ucr','uea','speechcommands','gtzan'],
               help="dataset source family")
p.add_argument('--ucr_name', type=str, default='FordA',
               help="UCR/UEA dataset name (ignored if source is speechcommands/gtzan)")
p.add_argument('--data_root', type=str, default='./data',
               help="root dir for torchaudio datasets (SC/GTZAN)")
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
    choices=['none', 'lpsconv', 'lpsconv_plus', 'spectral', 'sincnet_bank'],
    help="Prefilter: 'lpsconv' (old), 'lpsconv_plus', 'spectral', 'sincnet_bank', or 'none'.")

# your method (lpsconv) options
p.add_argument('--sym_kernel', type=int, default=21, help='odd kernel length for symmetric FIR')
p.add_argument('--sym_h', type=float, default=1.0)
p.add_argument('--sym_causal', action='store_true')
p.add_argument('--sym_no_residual', action='store_true')

# spectral pooling option
p.add_argument('--spec_cut', type=float, default=0.5, help='keep ratio in rFFT bins (0,1]')

# front-end knobs
p.add_argument('--fe_k', type=int, default=63, help='generic kernel length for new FEs (odd)')
p.add_argument('--fe_bands', type=int, default=8, help='bands for sincnet_bank')

args = p.parse_args()

# ---------------- setup ----------------
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

# ---------------- data ----------------
if args.source in ('ucr','uea'):
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_ucr_numpy3d(name=args.ucr_name, seed=args.seed)
    dataset_label = f"{args.source.upper()}:{args.ucr_name}"
elif args.source == 'speechcommands':
    X_tr, y_tr, X_va, y_va, X_te, y_te, labels = load_speechcommands(root=args.data_root, sample_rate=16000, seconds=1.0)
    dataset_label = "SPEECHCOMMANDS"
elif args.source == 'gtzan':
    X_tr, y_tr, X_va, y_va, X_te, y_te, labels = load_gtzan(root=args.data_root, sample_rate=16000, seconds=5.0)
    dataset_label = "GTZAN"
else:
    raise ValueError("unknown source")

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
print(f"[{dataset_label}] train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  "
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
elif args.front_end == 'lpsconv_plus':
    from TCN.common.front_end_factory import build_front_end
    front = build_front_end(kind='lpsconv_plus', in_channels=in_channels, k=args.fe_k)
    model = nn.Sequential(front, core)
elif args.front_end == 'spectral':
    from TCN.common.front_end_factory import build_front_end
    front = build_front_end(kind='spectral', in_channels=in_channels, cutoff_ratio=args.spec_cut)
    model = nn.Sequential(front, core)
elif args.front_end == 'sincnet_bank':
    from TCN.common.front_end_factory import build_front_end
    front = build_front_end(kind='sincnet_bank', in_channels=in_channels, k=args.fe_k, bands=args.fe_bands)
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
