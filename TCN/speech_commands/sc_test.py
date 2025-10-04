import argparse, torch
from speech_commands.data import make_loaders
from speech_commands.models import KWS_TCN
from speech_commands.utils import seed_all, run_epoch
from TCN.common.hartley_tcn import HartleyTCN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--channels', type=int, nargs='+', default=[128,128,128])
    ap.add_argument('--kernel_size', type=int, default=3)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--n_mels', type=int, default=40)
    ap.add_argument('--max_frames', type=int, default=200)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--sym', action='store_true',
                    help='enable symmetric front-end (default: off)')
    ap.add_argument('--sym_kernel', type=int, default=9,
                    help='symmetric conv kernel size (default: 9)')
    ap.add_argument('--sym_h', type=float, default=1.0,
                    help='Hartley mixing coefficient h (default: 1.0)')
    ap.add_argument('--sym_causal', action='store_true',
                    help='use causal symmetric conv (default: off)')
    ap.add_argument('--sym_no_residual', action='store_true',
                    help='disable residual add; use pure front output')
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tr, va, te, n_classes, n_mels = make_loaders(batch_size=args.batch_size,
                                                 n_mels=args.n_mels, max_frames=args.max_frames)
    base = KWS_TCN(n_mels=n_mels, channels=args.channels,
                   kernel_size=args.kernel_size, dropout=args.dropout,
                   num_classes=n_classes)
    # Wrap with HartleyTCN (no-op unless --sym is passed).
    model = HartleyTCN(
        base_tcn=base,
        in_channels=n_mels,                 # match input feature channels
        use_front=args.sym,
        k=args.sym_kernel,
        h=args.sym_h,
        causal=args.sym_causal,
        residual=(not args.sym_no_residual),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = 0.0
    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc = run_epoch(model, tr, opt, device, train=True)
        va_loss, va_acc = run_epoch(model, va, opt, device, train=False)
        print(f'[ep {ep:02d}] train_acc={tr_acc:.4f} val_acc={va_acc:.4f}')
        if va_acc > best: best = va_acc
    te_loss, te_acc = run_epoch(model, te, None, device, train=False)
    print(f'BEST_VAL_ACC={best:.4f}  TEST_ACC={te_acc:.4f}')

if __name__ == '__main__':
    main()
