# TCN/char_cnn/char_cnn_test.py
import argparse, sys, time, math, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim

# allow "from TCN.... import ..." when running from subdir
sys.path.append("../../")

from TCN.char_cnn.utils import *
from TCN.char_cnn.model import TCN
from TCN.common.front_end_factory import build_front_end  # spectral front-end
from TCN.common.hartley_tcn import HartleyTCN            # your method

parser = argparse.ArgumentParser(description='Sequence Modeling - Character Level Language Model')

# ---------------- core args (unchanged) ----------------
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--cuda', action='store_false', help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--emb_dropout', type=float, default=0.1)
parser.add_argument('--clip', type=float, default=0.15)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--ksize', type=int, default=3)
parser.add_argument('--levels', type=int, default=3)
parser.add_argument('--log_interval', type=int, default=100)     # <- underscore (not "log-interval")
parser.add_argument('--lr', type=float, default=4)
parser.add_argument('--emsize', type=int, default=100)
parser.add_argument('--optim', type=str, default='SGD')
parser.add_argument('--nhid', type=int, default=450)
parser.add_argument('--validseqlen', type=int, default=320)
parser.add_argument('--seq_len', type=int, default=400)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--dataset', type=str, default='ptb')

# ---------------- comparison: your method vs spectral pooling ----------------
parser.add_argument('--front_end', type=str, default='none',
                    choices=['none', 'lpsconv', 'spectral'],
                    help="Input front-end: 'lpsconv' (your linear-phase symmetric), 'spectral' (SpectralPool1d), or 'none'.")
# your method (lpsconv) toggles
parser.add_argument('--sym_kernel', type=int, default=9, help='odd kernel length for symmetric front-end')
parser.add_argument('--sym_h', type=float, default=1.0, help='scaling (if used in your SymmetricConv1d)')
parser.add_argument('--sym_causal', action='store_true', help='causal front-end (left padding) for LM')
parser.add_argument('--sym_no_residual', action='store_true', help='use front-end output only (no residual add)')

# spectral pooling toggle
parser.add_argument('--spec_cut', type=float, default=0.5,
                    help='SpectralPool keep ratio in rFFT bins (0,1], e.g., 0.45 keeps ~45% of Nyquist band')

args = parser.parse_args()

# seeds / cuda
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(args)
file, file_len, valfile, valfile_len, testfile, testfile_len, corpus = data_generator(args)
n_characters = len(corpus.dict)
train_data = batchify(char_tensor(corpus, file), args.batch_size, args)
val_data   = batchify(char_tensor(corpus, valfile), 1, args)
test_data  = batchify(char_tensor(corpus, testfile), 1, args)
print("Corpus size:", n_characters)

# model config
num_chans   = [args.nhid] * (args.levels - 1) + [args.emsize]
k_size      = args.ksize
dropout     = args.dropout
emb_dropout = args.emb_dropout

# 1) Build the original char-CNN TCN as usual
core = TCN(args.emsize, n_characters, num_chans,
           kernel_size=k_size, dropout=dropout, emb_dropout=emb_dropout)

# 2) Prepend the chosen front-end
#    - 'lpsconv': wrap with your HartleyTCN (linear-phase symmetric Conv1D)
#    - 'spectral': add SpectralPool1d right before TemporalConvNet
if args.front_end == 'lpsconv':
    model = HartleyTCN(
        base_tcn=core,
        in_channels=args.emsize,
        use_front=True,
        k=args.sym_kernel,
        h=args.sym_h,
        causal=args.sym_causal,
        residual=(not args.sym_no_residual)
    )
elif args.front_end == 'spectral':
    # NOTE: Spectral pooling is non-causal (global FFT); leaks future context for LM
    if args.dataset.lower() in {'ptb', 'ptb_char', 'char'}:
        print("[WARN] 'spectral' is non-causal and will leak future context in LM; use for diagnostics only.")
    front = build_front_end(kind='spectral',
                            in_channels=args.emsize,
                            cutoff_ratio=args.spec_cut)
    if hasattr(core, "tcn"):
        core.tcn = nn.Sequential(front, core.tcn)
        model = core
    else:
        raise AttributeError("Expected core.tcn to exist; update attribute name if different.")
else:
    model = core  # no front-end

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

def evaluate(source):
    model.eval()
    total_loss, count = 0.0, 0
    source_len = source.size(1)
    with torch.no_grad():
        for batch, i in enumerate(range(0, source_len - 1, args.validseqlen)):
            if i + args.seq_len - args.validseqlen >= source_len:
                continue
            inp, target = get_batch(source, i, args)
            output = model(inp)
            eff_history = args.seq_len - args.validseqlen
            final_output = output[:, eff_history:].contiguous().view(-1, n_characters)
            final_target = target[:, eff_history:].contiguous().view(-1)
            loss = criterion(final_output, final_target)
            total_loss += loss.data * final_output.size(0)
            count += final_output.size(0)
    return (total_loss.item() / max(count, 1))

def train(epoch):
    global lr
    model.train()
    total_loss, start_time, losses = 0.0, time.time(), []
    source, source_len = train_data, train_data.size(1)
    for batch_idx, i in enumerate(range(0, source_len - 1, args.validseqlen)):
        if i + args.seq_len - args.validseqlen >= source_len:
            continue
        inp, target = get_batch(source, i, args)
        optimizer.zero_grad()
        output = model(inp)
        eff_history = args.seq_len - args.validseqlen
        final_output = output[:, eff_history:].contiguous().view(-1, n_characters)
        final_target = target[:, eff_history:].contiguous().view(-1)
        loss = criterion(final_output, final_target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:   # <- fixed underscore
            cur_loss = total_loss / args.log_interval
            losses.append(cur_loss)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                  'loss {:5.3f} | bpc {:5.3f}'.format(
                epoch, batch_idx, int((source_len-0.5) / args.validseqlen), lr,
                elapsed * 1000 / args.log_interval, cur_loss, cur_loss / math.log(2)))
            total_loss = 0.0
            start_time = time.time()
    return sum(losses) / max(1, len(losses))

def main():
    global lr
    try:
        print(f"Training for {args.epochs} epochs...")
        all_losses, best_vloss = [], 1e9
        for epoch in range(1, args.epochs + 1):
            _ = train(epoch)
            vloss = evaluate(val_data)
            print('-' * 89)
            print('| End of epoch {:3d} | valid loss {:5.3f} | valid bpc {:8.3f}'.format(
                epoch, vloss, vloss / math.log(2)))
            test_loss = evaluate(test_data)
            print('=' * 89)
            print('| End of epoch {:3d} | test  loss {:5.3f} | test  bpc {:8.3f}'.format(
                epoch, test_loss, test_loss / math.log(2)))
            print('=' * 89)

            if epoch > 5 and len(all_losses) >= 3 and vloss > max(all_losses[-3:]):
                lr = lr / 10.0
                for g in optimizer.param_groups: g['lr'] = lr
            all_losses.append(vloss)

            if vloss < best_vloss:
                print("Saving...")
                save(model)
                best_vloss = vloss
    except KeyboardInterrupt:
        print('-' * 89)
        print("Saving before quit...")
        save(model)

    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.3f} | test bpc {:8.3f}'.format(
        test_loss, test_loss / math.log(2)))
    print('=' * 89)

if __name__ == "__main__":
    main()
