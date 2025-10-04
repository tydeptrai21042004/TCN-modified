import random, numpy as np, torch
from tqdm import tqdm

def seed_all(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_epoch(model, loader, opt, device, train=True):
    crit = torch.nn.CrossEntropyLoss()
    total=correct=0; loss_sum=0.0
    if train: model.train()
    else:     model.eval()
    for X,y in tqdm(loader, leave=False):
        X,y = X.to(device), y.to(device)
        if train:
            opt.zero_grad()
        logits = model(X)
        loss = crit(logits, y)
        if train:
            loss.backward(); opt.step()
        loss_sum += float(loss) * y.size(0)
        pred = logits.argmax(1); total += y.size(0); correct += int((pred==y).sum())
    return loss_sum/total, correct/total
