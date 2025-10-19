# word_cnn_test_frontend.py  (drop-in for your current script)
import argparse, time, math, torch, torch.nn as nn, torch.optim as optim
import sys
sys.path.append("../../")
from TCN.word_cnn.utils import *
from TCN.word_cnn.model import *          # your TCN class (unchanged)
from TCN.common.hartley_tcn import HartleyTCN  # we won't call it directly for LM
# front-end bits
from TCN.common.symmetric_conv1d import SymmetricConv1d
try:
    from TCN.common.front_end_factory import build_front_end
except Exception:
    build_front_end = None
try:
    from TCN.common.frontends.lpsconv_plus import LPSConvPlus
except Exception:
    LPSConvPlus = None

parser = argparse.ArgumentParser(description='Sequence Modeling - Word-level Language Modeling')

# --- your existing args unchanged ---
parser.add_argument('--batch_size', type=int, default=16, metavar='N')
parser.add_argument('--cuda', action='store_false', help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.45)
parser.add_argument('--emb_dropout', type=float, default=0.25)
parser.add_argument('--clip', type=float, default=0.35)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--ksize', type=int, default=3)
parser.add_argument('--data', type=str, default='./data/penn')
parser.add_argument('--emsize', type=int, default=600)
parser.add_argument('--levels', type=int, default=4)
parser.add_argument('--log-interval', type=int, default=100, metavar='N')
parser.add_argument('--lr', type=float, default=4)
parser.add_argument('--nhid', type=int, default=600)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--tied', action='store_false', help='tie encoder-decoder weights (default: True)')
parser.add_argument('--optim', type=str, default='SGD')
parser.add_argument('--validseqlen', type=int, default=40)
parser.add_argument('--seq_len', type=int, default=80)
parser.add_argument('--corpus', action='store_true')

# --- your old "sym" knobs (we will reuse them to configure FE) ---
parser.add_argument('--sym', action='store_true', help='(legacy) enable symmetric front-end')
parser.add_argument('--sym_kernel', type=int, default=9, help='odd kernel length for FE')
parser.add_argument('--sym_h', type=float, default=1.0)
parser.add_argument('--sym_causal', action='store_true')
parser.add_argument('--sym_no_residual', action='store_true')

# --- NEW: requested flag ---
parser.add_argument('--front_end', type=str, default='none',
                    choices=['none', 'lpsconv', 'lpsconv_plus'],
                    help="optional front-end before the TCN (word-LM-safe; causal)")

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(args)
corpus = data_generator(args)
eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size, args)
val_data   = batchify(corpus.valid, eval_batch_size, args)
test_data  = batchify(corpus.test,  eval_batch_size, args)
n_words = len(corpus.dictionary)

# ---------------- base model (unchanged class) ----------------
num_chans = [args.nhid] * (args.levels - 1) + [args.emsize]
base_model = TCN(args.emsize, n_words, num_chans,
                 dropout=args.dropout, emb_dropout=args.emb_dropout,
                 kernel_size=args.ksize, tied_weights=args.tied)

device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

# ---------------- FE wrapper: keeps your class untouched ----------------
class LMWithFrontEnd(nn.Module):
    """
    Takes your existing TCN word-LM model and inserts a 1-D FE over embeddings:
        tokens -> encoder -> (B,L,Em) --T--> (B,Em,L)
              -> FE (causal) [+ residual] -> TCN -> T -> decoder -> (B,L,V)
    Weight tying is preserved since we reuse base.decoder and base.encoder.
    """
    def __init__(self, base, kind='lpsconv', k=9, causal=True, residual=True):
        super().__init__()
        self.base = base
        em = base.encoder.embedding_dim
        if kind == 'lpsconv':
            self.fe = SymmetricConv1d(em, em, kernel_size=k, h=args.sym_h,
                                      causal=causal, bias=False)
            self.use_res = residual
        elif kind == 'lpsconv_plus':
            if build_front_end is not None:
                self.fe = build_front_end(kind='lpsconv_plus', in_channels=em, k=k)
            elif LPSConvPlus is not None:
                self.fe = LPSConvPlus(in_ch=em, k1=k, k2=k, causal=causal, unity_dc=False)
            else:
                raise RuntimeError("lpsconv_plus requested but not available. "
                                   "Install TCN.common.frontends.lpsconv_plus or front_end_factory.")
            self.use_res = True
        else:
            raise ValueError("Unknown FE kind")

    def forward(self, input_ids):  # (B, L)
        # Same embedding/dropout as your model:
        emb = self.base.drop(self.base.encoder(input_ids))     # (B, L, Em)  PyTorch Embedding expects indices. :contentReference[oaicite:0]{index=0}
        x = emb.transpose(1, 2)                                # -> (B, Em, L) for Conv1d. :contentReference[oaicite:1]{index=1}
        y = self.fe(x)
        if self.use_res:
            x = x + y
        else:
            x = y
        y = self.base.tcn(x).transpose(1, 2)                   # back to (B, L, H)
        y = self.base.decoder(y)                               # (B, L, V)
        return y.contiguous()

# Choose model according to --front_end
if args.front_end == 'none':
    model = base_model                           # EXACTLY your existing architecture
elif args.front_end == 'lpsconv':
    model = LMWithFrontEnd(base_model, kind='lpsconv',
                           k=args.sym_kernel,
                           causal=True if not args.sym_causal else True,
                           residual=not args.sym_no_residual)
elif args.front_end == 'lpsconv_plus':
    model = LMWithFrontEnd(base_model, kind='lpsconv_plus',
                           k=args.sym_kernel,
                           causal=True,            # word-LM must be causal
                           residual=True)
else:
    raise ValueError("invalid --front_end")

model = model.to(device)

criterion = nn.CrossEntropyLoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

def evaluate(data_source):
    model.eval()
    total_loss, processed = 0.0, 0
    with torch.no_grad():  # inference should disable grad for memory/speed. :contentReference[oaicite:2]{index=2}
        for i in range(0, data_source.size(1) - 1, args.validseqlen):
            if i + args.seq_len - args.validseqlen >= data_source.size(1) - 1:
                continue
            data, targets = get_batch(data_source, i, args, evaluation=True)
            output = model(data)
            eff_history = args.seq_len - args.validseqlen
            final_output = output[:, eff_history:].contiguous().view(-1, n_words)
            final_target = targets[:, eff_history:].contiguous().view(-1)
            loss = criterion(final_output, final_target)
            total_loss += (data.size(1) - eff_history) * loss.item()
            processed  += (data.size(1) - eff_history)
    return total_loss / processed

def train():
    model.train()
    total_loss, start_time = 0.0, time.time()
    for batch_idx, i in enumerate(range(0, train_data.size(1) - 1, args.validseqlen)):
        if i + args.seq_len - args.validseqlen >= train_data.size(1) - 1:
            continue
        data, targets = get_batch(train_data, i, args)
        optimizer.zero_grad()
        output = model(data)
        eff_history = args.seq_len - args.validseqlen
        if eff_history < 0:
            raise ValueError("validseqlen must be <= seq_len")
        final_target = targets[:, eff_history:].contiguous().view(-1)
        final_output = output[:, eff_history:].contiguous().view(-1, n_words)
        loss = criterion(final_output, final_target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch_idx, train_data.size(1) // args.validseqlen, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss, start_time = 0.0, time.time()

if __name__ == "__main__":
    best_vloss, all_vloss = 1e8, []
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train()
            val_loss  = evaluate(val_data)
            test_loss = evaluate(test_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
                  epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
            print('| end of epoch {:3d} | time: {:5.2f}s |  test loss {:5.2f} |  test ppl {:8.2f}'.format(
                  epoch, (time.time() - epoch_start_time), test_loss, math.exp(test_loss)))
            print('-' * 89)
            if val_loss < best_vloss:
                with open("model.pt", 'wb') as f:
                    print('Save model!\n'); torch.save(model, f)
                best_vloss = val_loss
            if epoch > 5 and (len(all_vloss) >= 5 and val_loss >= max(all_vloss[-5:])):
                lr = lr / 2.
                for pg in optimizer.param_groups: pg['lr'] = lr
            all_vloss.append(val_loss)
    except KeyboardInterrupt:
        print('-' * 89); print('Exiting from training early')

    with open("model.pt", 'rb') as f:
        model = torch.load(f)
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
    print('=' * 89)
