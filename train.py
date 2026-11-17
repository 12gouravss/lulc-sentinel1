import os, json, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Water","Trees","Grassland","Cropland","Built-up","Bare/Sparse","Wetland"]
PALETTE = ["#1f78b4","#33a02c","#b2df8a","#fdbf6f","#e31a1c","#b15928","#6a3d9a"]
N = len(CLASSES)
MAX_EPOCHS = 80
PATIENCE   = 12
BATCH      = 16


def load_clean(path):
    """Load an npz and sanitise labels (NaN -> 0, round, clip to 0..N-1)."""
    d = np.load(path)
    X = d["X"].astype(np.float32)
    Y = np.clip(np.round(np.nan_to_num(d["Y"].astype(np.float32), nan=0.0)), 0, N-1).astype(np.uint8)
    return X, Y


class SARDataset(Dataset):
    def __init__(self, X, Y, channels, augment=False):
        self.X, self.Y, self.ch, self.augment = X, Y, channels, augment
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.X[i][..., self.ch].copy(); y = self.Y[i].copy()
        if self.augment:
            if np.random.rand() < 0.5: x, y = np.fliplr(x).copy(), np.fliplr(y).copy()
            if np.random.rand() < 0.5: x, y = np.flipud(x).copy(), np.flipud(y).copy()
            k = np.random.randint(4); x, y = np.rot90(x, k).copy(), np.rot90(y, k).copy()
        return torch.from_numpy(x).permute(2, 0, 1).float(), torch.from_numpy(y).long()


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=2, n_classes=7, base=64):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base);      self.d2 = DoubleConv(base, base*2)
        self.d3 = DoubleConv(base*2, base*4);   self.d4 = DoubleConv(base*4, base*8)
        self.pool = nn.MaxPool2d(2);            self.bott = DoubleConv(base*8, base*16)
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2); self.u4 = DoubleConv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2);  self.u3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2);  self.u2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2);    self.u1 = DoubleConv(base*2, base)
        self.out = nn.Conv2d(base, n_classes, 1)
    def forward(self, x):
        c1 = self.d1(x); c2 = self.d2(self.pool(c1)); c3 = self.d3(self.pool(c2)); c4 = self.d4(self.pool(c3))
        b = self.bott(self.pool(c4))
        x = self.u4(torch.cat([self.up4(b), c4], 1)); x = self.u3(torch.cat([self.up3(x), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1)); x = self.u1(torch.cat([self.up1(x), c1], 1))
        return self.out(x)


def dice_loss(logits, target, eps=1e-6):
    probs = F.softmax(logits, dim=1); t1h = F.one_hot(target, N).permute(0, 3, 1, 2).float()
    inter = (probs*t1h).sum((0, 2, 3)); union = probs.sum((0, 2, 3)) + t1h.sum((0, 2, 3))
    return 1 - ((2*inter + eps)/(union + eps)).mean()


def class_weights(Y):
    counts = np.bincount(Y.ravel(), minlength=N).astype(np.float64)
    w = counts.sum() / (N * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader):
    
    model.eval(); inter = np.zeros(N); union = np.zeros(N); correct = total = 0
    cm = np.zeros((N, N), np.int64)
    for x, y in loader:
        x = x.to(device); y = y.to(device); pred = model(x).argmax(1)
        correct += (pred == y).sum().item(); total += y.numel()
        for c in range(N):
            p = (pred == c); g = (y == c)
            inter[c] += (p & g).sum().item(); union[c] += (p | g).sum().item()
        t = y.cpu().numpy().ravel(); pp = pred.cpu().numpy().ravel()
        cm += np.bincount(N*t + pp, minlength=N*N).reshape(N, N)
    iou = inter/np.maximum(union, 1)
    return correct/total, iou.mean(), iou, cm



def train_variant(tag, Xtr, Ytr, Xva, Yva, Xte, Yte, channels):
    in_ch = len(channels)
    ckpt  = f"unet_{tag}.pth"       # best weights (the deliverable)
    state = f"state_{tag}.pt"       # full resume state
    resj  = f"results_{tag}.json"   # final test results
    done  = f"done_{tag}.flag"

    
    if os.path.exists(done) and os.path.exists(resj):
        r = json.load(open(resj))
        print(f"[{tag}] already complete (val mIoU {r['best_val_miou']:.3f}, "
              f"test OA {r['test_oa']:.3f} mIoU {r['test_miou']:.3f}) -- skipping", flush=True)
        return r

    tr_dl = DataLoader(SARDataset(Xtr, Ytr, channels, augment=True), batch_size=BATCH, shuffle=True)
    va_dl = DataLoader(SARDataset(Xva, Yva, channels), batch_size=BATCH)
    te_dl = DataLoader(SARDataset(Xte, Yte, channels), batch_size=BATCH)

    model = UNet(in_ch=in_ch, n_classes=N, base=64).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    w  = class_weights(Ytr).to(device)
    ce = nn.CrossEntropyLoss(weight=w)
    criterion = lambda lo, ta: ce(lo, ta) + dice_loss(lo, ta)

    start_ep, best, wait = 1, 0.0, 0
    hist = {"train_loss": [], "val_loss": [], "val_oa": [], "val_miou": [], "lr": []}


    if os.path.exists(state):                      
        st = torch.load(state, map_location=device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start_ep = st["epoch"] + 1; best = st["best"]; wait = st["wait"]; hist = st["hist"]
        print(f"[{tag}] RESUMED from epoch {st['epoch']} (best val mIoU {best:.3f}) -> continuing at {start_ep}", flush=True)
    elif os.path.exists(ckpt):                       # warm-start from best weights only
        model.load_state_dict(torch.load(ckpt, map_location=device))
        best = evaluate(model, va_dl)[1]
        print(f"[{tag}] warm-started from {ckpt} (val mIoU {best:.3f})", flush=True)

    print(f"[{tag}] device {device} | in_ch {in_ch} | params "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M | "
          f"train {len(Xtr)} val {len(Xva)} test {len(Xte)}", flush=True)
    print(f"[{tag}] class weights: {[f'{x:.2f}' for x in w.cpu().numpy()]}", flush=True)

    for ep in range(start_ep, MAX_EPOCHS + 1):
        model.train(); tl = 0
        for x, y in tr_dl:
            x = x.to(device); y = y.to(device); opt.zero_grad()
            loss = criterion(model(x), y); loss.backward(); opt.step(); tl += loss.item()
        tl /= len(tr_dl)

        model.eval(); vl = 0
        with torch.no_grad():
            for x, y in va_dl:
                x = x.to(device); y = y.to(device); vl += criterion(model(x), y).item()
        vl /= len(va_dl)
        oa, miou, _, _ = evaluate(model, va_dl)
        sched.step(miou); lr = opt.param_groups[0]["lr"]
        for k, v in zip(hist, [tl, vl, oa, miou, lr]): hist[k].append(v)

        flag = ""
        if miou > best:
            best = miou; wait = 0; torch.save(model.state_dict(), ckpt); flag = " *"
        else:
            wait += 1
        # save resume state EVERY epoch (cull-proof)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best": best, "wait": wait, "hist": hist}, state)
        print(f"[{tag}] ep {ep:02d} | train {tl:.3f} | val {vl:.3f} | OA {oa:.3f} | mIoU {miou:.3f} | lr {lr:.1e}{flag}", flush=True)
        if wait >= PATIENCE:
            print(f"[{tag}] early stop at epoch {ep} (no val improvement for {PATIENCE})", flush=True)
            break

    # --- final test with the best checkpoint ---
    model.load_state_dict(torch.load(ckpt, map_location=device))
    oa, miou, per, cm = evaluate(model, te_dl)
    print(f"\n[{tag}] === TEST ===  OA {oa:.3f} | mIoU {miou:.3f}", flush=True)
    print(f"{'Class':12s} {'IoU':>6s} {'F1':>6s}", flush=True)
    perclass = {}
    for c in range(N):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        pr = tp/max(tp+fp, 1); re = tp/max(tp+fn, 1); f1 = 2*pr*re/max(pr+re, 1e-9)
        perclass[CLASSES[c]] = {"iou": float(per[c]), "f1": float(f1)}
        print(f"{CLASSES[c]:12s} {per[c]:6.3f} {f1:6.3f}", flush=True)

    res = {"tag": tag, "best_val_miou": float(best), "test_oa": float(oa),
           "test_miou": float(miou), "per_class": perclass, "history": hist}
    json.dump(res, open(resj, "w"))
    np.save(f"cm_{tag}.npy", cm)
    _plot_curves(tag, hist)
    if tag == "s1_lulc":
        _plot_confusion(tag, cm)
    open(done, "w").write("done")
    print(f"[{tag}] done -> {ckpt}, {resj}", flush=True)
    return res



def _plot_curves(tag, hist):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(ep, hist["train_loss"], label="train loss")
    ax[0].plot(ep, hist["val_loss"],   label="val loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title(f"{tag} — loss")
    ax[1].plot(ep, hist["val_oa"],   label="val OA")
    ax[1].plot(ep, hist["val_miou"], label="val mIoU")
    ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[1].set_title(f"{tag} — accuracy")
    plt.tight_layout(); plt.savefig("training_curves.png" if tag == "s1_lulc" else f"training_curves_{tag}.png", dpi=120)
    plt.close()


def _plot_confusion(tag, cm):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix (row-normalised)")
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046); plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120); plt.close()


def _plot_class_distribution(Ytr):
    counts = np.bincount(Ytr.ravel(), minlength=N); tot = counts.sum()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(N), 100*counts/tot, color=PALETTE)
    ax.set_xticks(range(N)); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_ylabel("% of pixels"); ax.set_title("Training class distribution")
    for i, c in enumerate(counts):
        ax.text(i, 100*c/tot, f"{100*c/tot:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); plt.savefig("class_distribution.png", dpi=120); plt.close()


def main():
    Xtr_all, Ytr_all = load_clean("train.npz")
    Xte, Yte = load_clean("test.npz")
    print(f"Loaded train {Xtr_all.shape} test {Xte.shape}", flush=True)
    print(f"Label values present: {np.unique(Ytr_all)}", flush=True)

    # deterministic 15% validation split
    rng = np.random.RandomState(0); idx = rng.permutation(len(Xtr_all))
    nval = int(0.15*len(idx)); vi, ti = idx[:nval], idx[nval:]
    Xtr, Ytr = Xtr_all[ti], Ytr_all[ti]
    Xva, Yva = Xtr_all[vi], Ytr_all[vi]
    print(f"Train {len(Xtr)} | Val {len(Xva)} | Test {len(Xte)}", flush=True)

    counts = np.bincount(Ytr.ravel(), minlength=N); tot = counts.sum()
    print("\nCLASS DISTRIBUTION (train):", flush=True)
    for c in range(N):
        print(f"  {CLASSES[c]:12s} {counts[c]:12,d} ({100*counts[c]/tot:5.1f}%)", flush=True)
    _plot_class_distribution(Ytr)

    print("\n=== MAIN model (VV+VH) ===", flush=True)
    main_res = train_variant("s1_lulc", Xtr, Ytr, Xva, Yva, Xte, Yte, channels=[0, 1])

    print("\n=== VV-only variant ===", flush=True)
    vv_res = train_variant("vv_only", Xtr, Ytr, Xva, Yva, Xte, Yte, channels=[0])

    print("\n MODEL COMPARISON ", flush=True)
    print(f"{'Model':18s} {'Test OA':>9s} {'Test mIoU':>10s}", flush=True)
    print(f"{'U-Net VV+VH':18s} {main_res['test_oa']:9.3f} {main_res['test_miou']:10.3f}", flush=True)
    print(f"{'U-Net VV-only':18s} {vv_res['test_oa']:9.3f} {vv_res['test_miou']:10.3f}", flush=True)
    print(flush=True)
    print("\nALL DONE. Deliverables in the Vault:", flush=True)
    print("  unet_s1_lulc.pth  unet_vv_only.pth", flush=True)
    print("  training_curves.png  confusion_matrix.png  class_distribution.png", flush=True)
    print("  results_s1_lulc.json  results_vv_only.json", flush=True)


if __name__ == "__main__":
    main()
