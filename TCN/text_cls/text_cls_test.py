# TCN/text_cls/textcls_test.py
import argparse, torch
from text_cls.data import make_loaders
from text_cls.models import TextTCN
from text_cls.utils import seed_everything, train_epoch, eval_epoch
from TCN.common.hartley_tcn import HartleyTCN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, required=True,
                    choices=['ag_news','imdb','yelp_polarity'])
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--embed_dim', type=int, default=100)
    ap.add_argument('--channels', type=int, nargs='+', default=[128,128,128])
    ap.add_argument('--kernel_size', type=int, default=3)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--max_len', type=int, default=512)
    ap.add_argument('--lr', type=float, default=2e-3)
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

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader, vocab, num_classes = make_loaders(
        args.dataset, batch_size=args.batch_size, max_len=args.max_len
    )

    base = TextTCN(
         vocab_size=len(vocab), embed_dim=args.embed_dim, num_classes=num_classes,
         channels=args.channels, kernel_size=args.kernel_size, dropout=args.dropout,
         padding_idx=vocab['<pad>']
)
    model = HartleyTCN(
        base_tcn=base,
        in_channels=args.embed_dim,          # match embedding dimension
        use_front=args.sym,
        k=args.sym_kernel,
        h=args.sym_h,
        causal=args.sym_causal,
        residual=(not args.sym_no_residual),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = 0.0
    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, device)
        te_loss, te_acc = eval_epoch(model, test_loader, device)
        print(f'[ep {ep:02d}] train_acc={tr_acc:.4f} test_acc={te_acc:.4f}')
        if te_acc > best: best = te_acc
    print(f'BEST_TEST_ACC={best:.4f}')

if __name__ == '__main__':
    main()
