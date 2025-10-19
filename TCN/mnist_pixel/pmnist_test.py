# pmnist_frontend_test.py
import argparse, sys, time, math, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Make top-level TCN importable (repo root two levels up)
sys.path.append("../../")

from TCN.mnist_pixel.utils import data_generator
from TCN.mnist_pixel.model import TCN as BaseTCN  # (your model below matches this)
from TCN.common.hartley_tcn import HartleyTCN     # linear-phase FE wrapper

# Optional factory / direct import for LPSConvPlus
try:
    from TCN.common.front_end_factory import build_front_end
except Exception:
    build_front_end = None
try:
    from TCN.common.frontends.lpsconv_plus import LPSConvPlus  # if you have this file
except Exception:
    LPSConvPlus = None

# ------------ CLI ------------
p = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST with optional front-ends')

# data / training
p.add_argument('--batch_size', type=int, default=64)
p.add_argument('--epochs', type=int, default=20)
p.add_argument('--seed', type=int, default=1111)
p.add_argument('--lr', type=float, default=2e-3)
p.add_argument('--optim', type=str, default='Adam')
p.add_argument('--clip', type=float, default=-1, help='grad clip; -1 = off')
p.add_argument('--log_interval', type=int, default=100)

# model core (TCN)
p.add_argument('--ksize', type=int, default=7)
p.add_argument('--levels', type=int, default=8)
p.add_argument('--nhid', type=int, default=25)
p.add_argument('--dropout', type=float, default=0.05)

# CUDA (note: keep legacy behavior—passing --cuda=False == CPU)
p.add_argument('--cuda', action='store_false', help='use CUDA (default: True)')

# task flags
p.add_argument('--permute', action='store_true', help='use permuted MNIST')

# NEW: front-end selector (your request)
p.add_argument('--front_end', type=str, default='none',
               choices=['none', 'lpsconv', 'lpsconv_plus'],
               help="linear-phase front-ends before TCN")
p.add_argument('--fe_k', type=int, default=9, help='odd kernel length for FE')
p.add_argument('--fe_causal', action='store_true', help='make FE causal (left-padded)')
p.add_argument('--fe_no_residual', action='store_true', help='(for lpsconv) disable residual add')

# Back-compat (kept; ignored if --front_end != none)
p.add_argument('--sym', action='store_true', help='(legacy) enable Hartley symmetric FE')
p.add_argument('--sym_kernel', type=int, default=9)
p.add_argument('--sym_h', type=float, default=1.0)
p.add_argument('--sym_causal', action='store_true')
p.add_argument('--sym_no_residual', action='store_true')

args = p.parse_args()
torch.manual_seed(args.seed)

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# ------------ Data ------------
root = './data/mnist'
batch_size = args.batch_size
n_classes = 10
input_channels = 1
seq_length = 784  # 28*28 flattened
epochs = args.epochs
steps = 0

train_loader, test_loader = data_generator(root, batch_size)
permute = torch.tensor(np.random.permutation(seq_length), dtype=torch.long)

# ------------ Model ------------
channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
base = BaseTCN(input_channels, n_classes, channel_sizes,
               kernel_size=kernel_size, dropout=args.dropout)

# Helper to build lpsconv_plus when no factory is available
def _build_lpsconv_plus(in_ch: int, k: int, causal: bool):
    if build_front_end is not None:
        return build_front_end(kind='lpsconv_plus', in_channels=in_ch, k=k)
    if LPSConvPlus is None:
        raise RuntimeError("LPSConvPlus not found. Install TCN.common.frontends.lpsconv_plus or provide front_end_factory.")
    return LPSConvPlus(in_ch=in_ch, k1=k, k2=k, causal=causal, unity_dc=True)

# Wire front-end per flag
if args.front_end == 'lpsconv':
    model = HartleyTCN(
        base_tcn=base,
        in_channels=input_channels,
        use_front=True,
        k=args.fe_k,
        h=1.0,
        causal=args.fe_causal,
        residual=not args.fe_no_residual
    )
elif args.front_end == 'lpsconv_plus':
    front = _build_lpsconv_plus(in_ch=input_channels, k=args.fe_k, causal=args.fe_causal)
    model = nn.Sequential(front, base)
else:
    # 'none' — keep legacy --sym* for backward compatibility if requested
    if args.sym:
        model = HartleyTCN(
            base_tcn=base,
            in_channels=input_channels,
            use_front=True,
            k=args.sym_kernel,
            h=args.sym_h,
            causal=args.sym_causal,
            residual=not args.sym_no_residual
        )
    else:
        model = base

device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
model.to(device)
permute = permute.to(device)

# ------------ Optimizer ------------
optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)

# ------------ Train / Test ------------
def train(ep):
    global steps
    model.train()
    running = 0.0
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, input_channels, seq_length)
        if args.permute:
            data = data.index_select(2, permute)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        running += loss.item()
        steps += seq_length
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {ep} [{batch_idx * batch_size}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {running/args.log_interval:.6f}\tSteps: {steps}')
            running = 0.0

@torch.no_grad()
def test():
    model.eval()
    total_loss = 0.0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(-1, input_channels, seq_length)
        if args.permute:
            data = data.index_select(2, permute)
        output = model(data)
        # reduction='sum' replaces deprecated size_average=False
        total_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(1)
        correct += (pred == target).sum().item()

    total_loss /= len(test_loader.dataset)
    acc = 100.0 * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {total_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.0f}%)\n')
    return total_loss, acc

if __name__ == "__main__":
    print(f"Front-end: {args.front_end} | fe_k={args.fe_k} | fe_causal={args.fe_causal} | fe_no_residual={args.fe_no_residual}")
    lr = args.lr
    for epoch in range(1, epochs + 1):
        train(epoch)
        test()
        if epoch % 10 == 0:
            lr /= 10
            for g in optimizer.param_groups:
                g['lr'] = lr
