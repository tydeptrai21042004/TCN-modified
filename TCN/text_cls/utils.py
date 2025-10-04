# TCN/text_cls/utils.py
import random, os, numpy as np, torch, torch.nn as nn
from tqdm import tqdm

def seed_everything(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, loader, optimizer, device):
    model.train(); crit = torch.nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for X, y in tqdm(loader, leave=False):
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = crit(logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        loss_sum += float(loss) * y.size(0)
        pred = logits.argmax(1); total += y.size(0); correct += int((pred==y).sum())
    return loss_sum/total, correct/total

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval(); crit = torch.nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = crit(logits, y)
        loss_sum += float(loss) * y.size(0)
        pred = logits.argmax(1); total += y.size(0); correct += int((pred==y).sum())
    return loss_sum/total, correct/total
