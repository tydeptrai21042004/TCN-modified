import os, torch
from torch.utils.data import DataLoader
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

def _labels_from_subset(root, subset):
    ds = SPEECHCOMMANDS(root, download=True, subset=subset)
    labels = sorted({ds[i][2] for i in range(len(ds))})
    lab2id = {lab:i for i,lab in enumerate(labels)}
    return labels, lab2id

class _Wrap(torch.utils.data.Dataset):
    def __init__(self, root, subset="training", sample_rate=16000, n_mels=40, max_frames=200):
        self.ds = SPEECHCOMMANDS(root, download=True, subset=subset)
        self.labels, self.lab2id = _labels_from_subset(root, "training")
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        self.amplog = torchaudio.transforms.AmplitudeToDB()
        self.max_frames = max_frames

    def __len__(self): return len(self.ds)

    def __getitem__(self, i):
        wav, sr, label, *_ = self.ds[i]
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        spec = self.amplog(self.mel(wav)).squeeze(0)  # (n_mels, T)
        T = spec.size(1)
        if T >= self.max_frames: spec = spec[:, :self.max_frames]
        else:                    spec = torch.nn.functional.pad(spec, (0, self.max_frames - T))
        y = self.lab2id[label]
        return spec.float(), torch.tensor(y, dtype=torch.long)

def make_loaders(root="./data", batch_size=128, n_mels=40, max_frames=200):
    train = _Wrap(root, "training", n_mels=n_mels, max_frames=max_frames)
    valid = _Wrap(root, "validation", n_mels=n_mels, max_frames=max_frames)
    test  = _Wrap(root, "testing",   n_mels=n_mels, max_frames=max_frames)
    def _mk(ds): return DataLoader(ds, batch_size=batch_size, shuffle=(ds is train), num_workers=0)
    return _mk(train), _mk(valid), _mk(test), len(train.lab2id), n_mels
