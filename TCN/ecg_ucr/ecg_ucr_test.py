import argparse, sys, time, math, warnings, os
warnings.filterwarnings("ignore")
from TCN.common.hartley_tcn import HartleyTCN             # your linear-phase front-end wrapper
from TCN.common.front_end_factory import build_front_end   # now supports multiple FEs

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

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

# ---------------- misc utils ----------------
def _zscore_train_stats(Xtr):
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
    return mu, sd

def _apply_zscore(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)

class NumpyTSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

# ---------------- dataset loaders ----------------
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

# ---- streaming/lazy audio datasets to avoid OOM ----
def _labels_from_walker_paths(paths):
    # Extract folder name just above the file
    labs = set()
    for p in paths:
        p = os.path.normpath(p)
        parts = p.split(os.sep)
        if len(parts) >= 2:
            labs.add(parts[-2])
    return sorted(labs)

class _FixedLenAudioDataset(Dataset):
    """Wrap a torchaudio dataset to (C=1, T=target_len) with per-sample z-score.
    Assumes underlying dataset returns (waveform, sample_rate, label, ...).
    """
    def __init__(self, base_ds, labels, sample_rate, target_len, resampler):
        self.base = base_ds
        self.labels = labels
        self.lab2id = {lab: i for i, lab in enumerate(labels)}
        self.sample_rate = sample_rate
        self.target_len = int(target_len)
        self.resampler = resampler
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        item = self.base[idx]
        # torchaudio returns different tuples per dataset; label is usually at index 2
        waveform, sr, label = item[0], item[1], item[2]
        if sr != self.sample_rate and self.resampler is not None:
            waveform = self.resampler(waveform)
        wav = waveform.mean(dim=0, keepdim=True)  # mono -> (1,T)
        T = wav.shape[-1]
        if T < self.target_len:
            wav = torch.nn.functional.pad(wav, (0, self.target_len - T))
        else:
            wav = wav[..., :self.target_len]
        # per-sample standardization
        m = wav.mean(dim=-1, keepdim=True)
        s = wav.std(dim=-1, keepdim=True).clamp_min(1e-8)
        wav = (wav - m) / s
        y = torch.tensor(self.lab2id[label], dtype=torch.long)
        return wav, y

def build_speechcommands_datasets(root: str, sample_rate=16000, seconds=1.0):
    import torchaudio
    from torchaudio.datasets import SPEECHCOMMANDS

    class SubsetSC(SPEECHCOMMANDS):
        def __init__(self, subset: str, **kw):
            super().__init__(**kw)
            def load_list(filename):
                filepath = os.path.join(self._path, filename)
                with open(filepath) as f:
                    return [os.path.join(self._path, l.strip()) for l in f]
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
    # SC v2 is 16 kHz; keep a resampler in case user changes sample_rate
    resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=sample_rate) if sample_rate != 16000 else None

    train_base = SubsetSC(subset="training",   root=root, download=True)
    valid_base = SubsetSC(subset="validation", root=root, download=True)
    test_base  = SubsetSC(subset="testing",    root=root, download=True)

    labels = _labels_from_walker_paths(train_base._walker)
    tr = _FixedLenAudioDataset(train_base, labels, sample_rate, target_len, resample)
    va = _FixedLenAudioDataset(valid_base, labels, sample_rate, target_len, resample)
    te = _FixedLenAudioDataset(test_base,  labels, sample_rate, target_len, resample)
    return tr, va, te, labels

def build_gtzan_datasets(root: str, sample_rate=16000, seconds=5.0):
    import torchaudio
    from torchaudio.datasets import GTZAN
    target_len = int(sample_rate * seconds)
    resample = torchaudio.transforms.Resample(orig_freq=22050, new_freq=sample_rate) if sample_rate != 22050 else None

    def _make(base):
        labels = _labels_from_walker_paths(base._walker)
        tr = _FixedLenAudioDataset(base if getattr(base, 'subset', None) == 'training' else GTZAN(root=root, subset='training', download=False), labels, sample_rate, target_len, resample)
        va = _FixedLenAudioDataset(base if getattr(base, 'subset', None) == 'validation' else GTZAN(root=root, subset='validation', download=False), labels, sample_rate, target_len, resample)
        te = _FixedLenAudioDataset(base if getattr(base, 'subset', None) == 'testing' else GTZAN(root=root, subset='testing', download=False), labels, sample_rate, target_len, resample)
        return tr, va, te, labels

    # Try offline first
    try:
        base = GTZAN(root=root, subset='training', download=False)
        return _make(base)
    except Exception:
        # Fallback to download; if it fails, raise a helpful error
        try:
            train_base = GTZAN(root=root, subset='training',   download=True)
            valid_base = GTZAN(root=root, subset='validation', download=True)
            test_base  = GTZAN(root=root, subset='testing',    download=True)
            labels = _labels_from_walker_paths(train_base._walker)
            tr = _FixedLenAudioDataset(train_base, labels, sample_rate, target_len, resample)
            va = _FixedLenAudioDataset(valid_base, labels, sample_rate, target_len, resample)
            te = _FixedLenAudioDataset(test_base,  labels, sample_rate, target_len, resample)
            return tr, va, te, labels
        except Exception as e:
            raise RuntimeError(f"GTZAN not found under '{root}'. Network download failed ({e})."
                               f"→ Manually place the dataset under '{root}' (folder 'gtzan' with cached files) or switch to a local dataset like 'esc50_local' or 'urban8k_local'.")

class _SubsetView(Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = list(indices)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        return self.base[self.indices[i]]

def build_yesno_datasets(root: str, sample_rate=8000, seconds=1.0):
    """YESNO (8-bit yes/no sequence) → map 8 bits to a categorical class.
    Splits: 60%/20%/20% (stratified by pattern id)."""
    import torchaudio
    from torchaudio.datasets import YESNO
    from sklearn.model_selection import StratifiedShuffleSplit

    base = YESNO(root=root, download=True)

    # Build labels (binary pattern 8 bits → int)
    ids = list(range(len(base)))
    lab_ids = []
    for i in ids:
        _, sr, bits = base[i]
        lab_ids.append(int(''.join(str(int(b)) for b in bits), 2))

    # Stratified 60/20/20
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=123)
    (train_idx, tmp_idx), = sss1.split(ids, lab_ids)
    tmp_y = [lab_ids[i] for i in tmp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=123)
    (val_rel, test_rel), = sss2.split(list(range(len(tmp_idx))), tmp_y)
    val_idx  = [tmp_idx[i] for i in val_rel]
    test_idx = [tmp_idx[i] for i in test_rel]

    # Map used classes to contiguous ids
    uniq = sorted(set(lab_ids))
    lab2id = {lab:i for i,lab in enumerate(uniq)}

    target_len = int(sample_rate * seconds)
    resample = torchaudio.transforms.Resample(orig_freq=8000, new_freq=sample_rate) if sample_rate != 8000 else None

    class YesNoFixed(Dataset):
        def __init__(self, base, indices, lab2id, sample_rate, target_len, resampler):
            self.base=base; self.indices=list(indices); self.lab2id=lab2id
            self.sample_rate=sample_rate; self.target_len=int(target_len); self.resampler=resampler
        def __len__(self): return len(self.indices)
        def __getitem__(self, j):
            i = self.indices[j]
            waveform, sr, bits = self.base[i]
            if sr != self.sample_rate and self.resampler is not None:
                waveform = self.resampler(waveform)
            wav = waveform.mean(dim=0, keepdim=True)
            T = wav.shape[-1]
            if T < self.target_len: wav = torch.nn.functional.pad(wav, (0, self.target_len - T))
            else: wav = wav[..., :self.target_len]
            m = wav.mean(dim=-1, keepdim=True); s = wav.std(dim=-1, keepdim=True).clamp_min(1e-8); wav = (wav-m)/s
            lab = int(''.join(str(int(b)) for b in bits), 2)
            y = torch.tensor(self.lab2id[lab], dtype=torch.long)
            return wav, y

    tr = YesNoFixed(base, train_idx, lab2id, sample_rate, target_len, resample)
    va = YesNoFixed(base, val_idx,   lab2id, sample_rate, target_len, resample)
    te = YesNoFixed(base, test_idx,  lab2id, sample_rate, target_len, resample)
    labels = [str(u) for u in uniq]
    return tr, va, te, labels

def build_vctk_sid_datasets(root: str, sample_rate=16000, seconds=2.0):
    """VCTK speaker-ID classification (labels are speaker IDs like 'p225').
    Splits: 80/10/10 stratified by speaker."""
    import torchaudio
    from torchaudio.datasets import VCTK
    from sklearn.model_selection import StratifiedShuffleSplit

    base = VCTK(root=root, download=True)

    # Extract speaker ids from dataset items (field at index 2 in torchaudio>=0.11)
    speakers = []
    for i in range(len(base)):
        item = base[i]
        spk = item[2] if len(item) > 2 else str(item[-1])
        speakers.append(str(spk))

    uniq = sorted(set(speakers))
    spk2id = {s:i for i,s in enumerate(uniq)}

    ids = list(range(len(base)))
    y = [spk2id[s] for s in speakers]

    # 80/20 then 50/50 of the 20% → 10/10
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
    (train_idx, tmp_idx), = sss1.split(ids, y)
    tmp_y = [y[i] for i in tmp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=123)
    (val_rel, test_rel), = sss2.split(list(range(len(tmp_idx))), tmp_y)
    val_idx  = [tmp_idx[i] for i in val_rel]
    test_idx = [tmp_idx[i] for i in test_rel]

    # resample VCTK 48 kHz → desired
    resample = torchaudio.transforms.Resample(orig_freq=48000, new_freq=sample_rate) if sample_rate != 48000 else None
    target_len = int(sample_rate * seconds)

    # Wrap each split with a subset view and our fixed-len adapter
    tr = _FixedLenAudioDataset(_SubsetView(base, train_idx), uniq, sample_rate, target_len, resample)
    va = _FixedLenAudioDataset(_SubsetView(base, val_idx),   uniq, sample_rate, target_len, resample)
    te = _FixedLenAudioDataset(_SubsetView(base, test_idx),  uniq, sample_rate, target_len, resample)
    return tr, va, te, uniq

def _assert_exists(p, msg):
    if not os.path.exists(p):
        raise RuntimeError(msg + f" — missing: {p}")

class AudioPathLabelDataset(Dataset):
    def __init__(self, items, labels, sample_rate, seconds, orig_sr=None):
        import torchaudio
        self.items = items  # list of (path, label_name)
        self.labels = labels
        self.lab2id = {lab:i for i, lab in enumerate(labels)}
        self.sr = int(sample_rate)
        self.T = int(self.sr * seconds)
        self.ta = torchaudio
        self.resampler = None if (orig_sr is None or orig_sr == self.sr) else self.ta.transforms.Resample(orig_freq=orig_sr, new_freq=self.sr)
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, lab = self.items[idx]
        wav, sr = self.ta.load(path)
        if sr != self.sr and self.resampler is not None:
            wav = self.resampler(wav)
        wav = wav.mean(dim=0, keepdim=True)
        T = wav.shape[-1]
        if T < self.T:
            wav = torch.nn.functional.pad(wav, (0, self.T - T))
        else:
            wav = wav[..., :self.T]
        m = wav.mean(dim=-1, keepdim=True); s = wav.std(dim=-1, keepdim=True).clamp_min(1e-8)
        wav = (wav - m) / s
        y = torch.tensor(self.lab2id[lab], dtype=torch.long)
        return wav, y

def build_esc50_local_datasets(root: str, sample_rate=16000, seconds=5.0):
    """ESC-50 local loader (no download). Expect structure:
    root/ESC-50/meta/esc50.csv and root/ESC-50/audio/*.wav
    Splits: folds 1-4 train, fold 5 test; 10% of train → val (stratified).
    """
    import csv
    base_dir = os.path.join(root, 'ESC-50')
    meta = os.path.join(base_dir, 'meta', 'esc50.csv')
    audio_dir = os.path.join(base_dir, 'audio')
    _assert_exists(meta, "ESC-50 meta CSV not found")
    _assert_exists(audio_dir, "ESC-50 audio folder not found")

    rows = []
    with open(meta, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((int(r['fold']), r['category'], os.path.join(audio_dir, r['filename'])))
    labels = sorted({r[1] for r in rows})
    items_train = [(p, c) for fold, c, p in rows if fold in (1,2,3,4)]
    items_test  = [(p, c) for fold, c, p in rows if fold == 5]

    # Stratified 10% of train → val
    from sklearn.model_selection import StratifiedShuffleSplit
    y = [c for _, c in items_train]
    idxs = list(range(len(items_train)))
    (tr_idx, va_idx), = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=123).split(idxs, y)
    tr_items = [items_train[i] for i in tr_idx]
    va_items = [items_train[i] for i in va_idx]

    tr = AudioPathLabelDataset(tr_items, labels, sample_rate, seconds, orig_sr=44100)
    va = AudioPathLabelDataset(va_items, labels, sample_rate, seconds, orig_sr=44100)
    te = AudioPathLabelDataset(items_test, labels, sample_rate, seconds, orig_sr=44100)
    return tr, va, te, labels

def build_urban8k_local_datasets(root: str, sample_rate=16000, seconds=4.0):
    """UrbanSound8K local loader (no download). Expect structure:
    root/UrbanSound8K/metadata/UrbanSound8K.csv
    root/UrbanSound8K/audio/foldX/*.wav
    Splits: folds 1-8 train, 9 val, 10 test.
    """
    import csv
    base_dir = os.path.join(root, 'UrbanSound8K')
    meta = os.path.join(base_dir, 'metadata', 'UrbanSound8K.csv')
    audio_root = os.path.join(base_dir, 'audio')
    _assert_exists(meta, "UrbanSound8K metadata CSV not found")
    _assert_exists(audio_root, "UrbanSound8K audio folder not found")

    items_train, items_val, items_test = [], [], []
    labels_set = set()
    with open(meta, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            fold = int(r['fold'])
            cls  = r['class']
            fname= r['slice_file_name']
            path = os.path.join(audio_root, f'fold{fold}', fname)
            labels_set.add(cls)
            dest = items_train if fold <= 8 else (items_val if fold == 9 else items_test)
            dest.append((path, cls))
    labels = sorted(labels_set)

    tr = AudioPathLabelDataset(items_train, labels, sample_rate, seconds, orig_sr=44100)
    va = AudioPathLabelDataset(items_val,   labels, sample_rate, seconds, orig_sr=44100)
    te = AudioPathLabelDataset(items_test,  labels, sample_rate, seconds, orig_sr=44100)
    return tr, va, te, labels

def build_physionet2017_local_datasets(root: str, sample_rate=300, seconds=30.0):
    """PhysioNet 2017 Challenge (AF detection) local loader.
    Expect: root/training2017/*.wav and root/REFERENCE.csv
    Split: 80/10/10 stratified.
    """
    import csv
    base = root
    if os.path.isdir(os.path.join(root, 'training2017')):
        base = root
    elif os.path.isdir(os.path.join(root, 'PhysioNet2017', 'training2017')):
        base = os.path.join(root, 'PhysioNet2017')
    train_dir = os.path.join(base, 'training2017')
    ref_csv   = os.path.join(base, 'REFERENCE.csv')
    _assert_exists(train_dir, "PhysioNet2017 training2017 folder not found")
    _assert_exists(ref_csv,   "PhysioNet2017 REFERENCE.csv not found")

    rows = []
    with open(ref_csv, 'r', newline='') as f:
        reader = csv.reader(f)
        for rid, lab in reader:
            wav = os.path.join(train_dir, rid + '.wav')
            if os.path.exists(wav):
                rows.append((wav, lab))
    # label mapping
    labels = sorted({lab for _, lab in rows})
    from sklearn.model_selection import StratifiedShuffleSplit
    y = [lab for _, lab in rows]
    idxs = list(range(len(rows)))
    (train_idx, tmp_idx), = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123).split(idxs, y)
    tmp_y = [y[i] for i in tmp_idx]
    (val_rel, test_rel), = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=123).split(list(range(len(tmp_idx))), tmp_y)
    val_idx  = [tmp_idx[i] for i in val_rel]
    test_idx = [tmp_idx[i] for i in test_rel]

    tr_items = [rows[i] for i in train_idx]
    va_items = [rows[i] for i in val_idx]
    te_items = [rows[i] for i in test_idx]

    tr = AudioPathLabelDataset(tr_items, labels, sample_rate, seconds, orig_sr=300)
    va = AudioPathLabelDataset(va_items, labels, sample_rate, seconds, orig_sr=300)
    te = AudioPathLabelDataset(te_items, labels, sample_rate, seconds, orig_sr=300)
    return tr, va, te, labels

def build_synthetic_sines_datasets(n_train=2000, n_val=400, n_test=400, sample_rate=100, seconds=2.0, n_classes=5, seed=123):
    rng = np.random.RandomState(seed)
    T = int(sample_rate * seconds)
    Xs, Ys = [], []
    freqs = np.linspace(1.0, 12.0, n_classes)
    def gen(n):
        X, y = [], []
        for i in range(n):
            c = rng.randint(0, n_classes)
            f = freqs[c] + rng.randn()*0.1
            t = np.arange(T) / sample_rate
            sig = np.sin(2*np.pi*f*t + rng.rand()*2*np.pi)
            sig += 0.1 * rng.randn(T)
            X.append(sig[None, :].astype(np.float32))
            y.append(c)
        return np.stack(X), np.array(y, dtype=np.int64)
    Xtr, ytr = gen(n_train); Xva, yva = gen(n_val); Xte, yte = gen(n_test)
    mu, sd = _zscore_train_stats(Xtr)
    return _apply_zscore(Xtr, mu, sd), ytr, _apply_zscore(Xva, mu, sd), yva, _apply_zscore(Xte, mu, sd), yte

# ---------------- argparse ----------------
p = argparse.ArgumentParser("TCN classification across multiple time-series datasets")
p.add_argument('--source', type=str, default='ucr',
               choices=['ucr','uea','speechcommands','gtzan','yesno','vctk_sid','esc50_local','urban8k_local','physionet2017_local','synthetic_sines'],
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
p.add_argument('--workers', type=int, default=2, help='DataLoader workers for streaming audio datasets')

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

# front-end knobs
p.add_argument('--fe_k', type=int, default=63, help='generic kernel length for new FEs (odd)')
p.add_argument('--fe_bands', type=int, default=8, help='bands for sincnet_bank')

args = p.parse_args()

# ---------------- setup ----------------
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

# ---------------- data ----------------
use_numpy_pipeline = args.source in ('ucr','uea')
labels = None

if use_numpy_pipeline:
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_ucr_numpy3d(name=args.ucr_name, seed=args.seed)
    dataset_label = f"{args.source.upper()}:{args.ucr_name}"

    in_channels = X_tr.shape[1]
    T = X_tr.shape[2]
    num_classes = int(max(y_tr.max(), y_va.max(), y_te.max()) + 1)

    train_loader = DataLoader(NumpyTSDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(NumpyTSDataset(X_va, y_va), batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(NumpyTSDataset(X_te, y_te), batch_size=args.batch_size, shuffle=False)
else:
    if args.source == 'speechcommands':
        tr_ds, va_ds, te_ds, labels = build_speechcommands_datasets(root=args.data_root, sample_rate=16000, seconds=1.0)
        dataset_label = "SPEECHCOMMANDS"
    elif args.source == 'gtzan':
        tr_ds, va_ds, te_ds, labels = build_gtzan_datasets(root=args.data_root, sample_rate=16000, seconds=5.0)
        dataset_label = "GTZAN"
    elif args.source == 'yesno':
        tr_ds, va_ds, te_ds, labels = build_yesno_datasets(root=args.data_root, sample_rate=8000, seconds=1.0)
        dataset_label = "YESNO"
    elif args.source == 'vctk_sid':
        tr_ds, va_ds, te_ds, labels = build_vctk_sid_datasets(root=args.data_root, sample_rate=16000, seconds=2.0)
        dataset_label = "VCTK-SID"
    elif args.source == 'esc50_local':
        tr_ds, va_ds, te_ds, labels = build_esc50_local_datasets(root=args.data_root, sample_rate=16000, seconds=5.0)
        dataset_label = "ESC-50 (local)"
    elif args.source == 'urban8k_local':
        tr_ds, va_ds, te_ds, labels = build_urban8k_local_datasets(root=args.data_root, sample_rate=16000, seconds=4.0)
        dataset_label = "UrbanSound8K (local)"
    elif args.source == 'physionet2017_local':
        tr_ds, va_ds, te_ds, labels = build_physionet2017_local_datasets(root=args.data_root, sample_rate=300, seconds=30.0)
        dataset_label = "PhysioNet2017 (local)"
    elif args.source == 'synthetic_sines':
        X_tr, y_tr, X_va, y_va, X_te, y_te = build_synthetic_sines_datasets()
        dataset_label = "Synthetic Sines"
        use_numpy_pipeline = True
    else:
        raise ValueError("unknown source")

    num_classes = len(labels)

    train_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # infer input dims from a single batch
    _xb, _yb = next(iter(train_loader))
    in_channels = int(_xb.shape[1])
    T = int(_xb.shape[-1])

steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)
print(f"[{dataset_label}] train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  "
      f"test={len(test_loader.dataset)}  | C={in_channels} T={T} classes={num_classes}  "
      f"| steps/epoch={steps_per_epoch}")

# ---------------- model ----------------
core = TCNClassifier(in_channels=in_channels, num_classes=num_classes,
                     levels=args.levels, hidden=args.hidden,
                     kernel_size=args.ksize, dropout=args.dropout)

if args.front_end == 'lpsconv':
    model = HartleyTCN(base_tcn=core, in_channels=in_channels,
                       use_front=True, k=args.sym_kernel, h=1.0,
                       causal=args.sym_causal, residual=not args.sym_no_residual)
elif args.front_end == 'lpsconv_plus':
    front = build_front_end(kind='lpsconv_plus', in_channels=in_channels, k=args.fe_k)
    model = nn.Sequential(front, core)
elif args.front_end == 'spectral':
    front = build_front_end(kind='spectral', in_channels=in_channels, cutoff_ratio=args.spec_cut)
    model = nn.Sequential(front, core)
elif args.front_end == 'sincnet_bank':
    front = build_front_end(kind='sincnet_bank', in_channels=in_channels, k=args.fe_k, bands=args.fe_bands)
    model = nn.Sequential(front, core)
elif args.front_end == 'fir_remez':
    front = build_front_end(kind='fir_remez', in_channels=in_channels, k=args.fe_k)
    model = nn.Sequential(front, core)
elif args.front_end == 'blurpool':
    front = build_front_end(kind='blurpool', in_channels=in_channels)
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
