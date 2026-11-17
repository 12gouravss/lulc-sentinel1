import os, json, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import rasterio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# reuse the exact model / data / metric code from the main trainer
from train import (UNet, SARDataset, dice_loss, class_weights, evaluate,
                   device, N, CLASSES)

torch.manual_seed(0); np.random.seed(0)

TARGET      = "catalonia"          # held-out region the source model never trained on
MONTHS      = [3, 6, 9]            # a few months to enough patches for a small data study
PATCH, STRIDE = 256, 128           # overlap to get more samples from one small region
EPOCHS      = 35
BATCH       = 8
PRETRAINED  = "unet_s1_lulc.pth"   # the source model (encoder we transfer)
TRAIN_FRACTION = 0.30              # fraction of the target "pool" used as the few labels


def patchify_tile(path):
    with rasterio.open(path) as r:
        arr = r.read().astype(np.float32)          # (3, H, W): VV, VH, label
    img = np.stack([arr[0], arr[1]], -1); lab = arr[2]
    H, W = lab.shape; X, Y = [], []
    for i in range(0, H - PATCH + 1, STRIDE):
        for j in range(0, W - PATCH + 1, STRIDE):
            xp = img[i:i+PATCH, j:j+PATCH]; yp = lab[i:i+PATCH, j:j+PATCH]
            if np.isnan(xp).mean() < 0.1 and not np.isnan(yp).any():
                X.append(np.nan_to_num(xp, nan=0.0).astype(np.float32))
                Y.append(np.clip(yp, 0, N-1).astype(np.uint8))
    return X, Y


def build_target():
    X, Y = [], []
    for m in MONTHS:
        p = f"tiles/{TARGET}_2021_{m:02d}.tif"
        if not os.path.exists(p):
            print(f"  (skip missing {p})", flush=True); continue
        xs, ys = patchify_tile(p); X += xs; Y += ys
    X = np.array(X); Y = np.array(Y)
    rng = np.random.RandomState(0); idx = rng.permutation(len(X))
    X, Y = X[idx], Y[idx]
    n_test = int(0.60 * len(X))
    Xte, Yte = X[:n_test], Y[:n_test]
    pool_X, pool_Y = X[n_test:], Y[n_test:]
    n_tr = max(BATCH, int(TRAIN_FRACTION * len(pool_X)))
    Xtr, Ytr = pool_X[:n_tr], pool_Y[:n_tr]
    return Xtr, Ytr, Xte, Yte


def make_model(pretrained=False, freeze_encoder=False):
    m = UNet(in_ch=2, n_classes=N, base=64).to(device)
    if pretrained:
        m.load_state_dict(torch.load(PRETRAINED, map_location=device))
    if freeze_encoder:
        for mod in [m.d1, m.d2, m.d3, m.d4, m.bott]:    
            for p in mod.parameters(): p.requires_grad = False
    return m


def train_regime(name, model, tr_dl, te_dl, w, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    ce = nn.CrossEntropyLoss(weight=w); crit = lambda lo, t: ce(lo, t) + dice_loss(lo, t)
    hist = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y in tr_dl:
            x = x.to(device); y = y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        oa, miou, _, _ = evaluate(model, te_dl); hist.append(miou)
        if ep % 5 == 0 or ep == 1:
            print(f"  [{name}] ep {ep:02d} | target-test OA {oa:.3f} | mIoU {miou:.3f}", flush=True)
    oa, miou, iou, _ = evaluate(model, te_dl)
    print(f"  [{name}] FINAL  OA {oa:.3f} | mIoU {miou:.3f}", flush=True)
    return {"name": name, "oa": oa, "miou": miou, "hist": hist}


def main():
    if not os.path.exists(PRETRAINED):
        print(f"ERROR: {PRETRAINED} not found — run train.py first."); return
    print(f"Transfer learning (Lecture 08) | target region: {TARGET} | device {device}", flush=True)
    Xtr, Ytr, Xte, Yte = build_target()
    print(f"Target labels used for adaptation: {len(Xtr)} patches | held-out target test: {len(Xte)} patches", flush=True)

    tr_dl = DataLoader(SARDataset(Xtr, Ytr, [0, 1], augment=True), batch_size=BATCH, shuffle=True)
    te_dl = DataLoader(SARDataset(Xte, Yte, [0, 1]), batch_size=BATCH)
    w = class_weights(Ytr).to(device)

    results = []

    # 1) source model, zero-shot (no target training)
    m0 = make_model(pretrained=True)
    oa, miou, _, _ = evaluate(m0, te_dl)
    print(f"  [zero-shot] source model, no adaptation  OA {oa:.3f} | mIoU {miou:.3f}", flush=True)
    results.append({"name": "Source (zero-shot)", "oa": oa, "miou": miou, "hist": None})

    # 2) from scratch
    results.append(train_regime("scratch", make_model(pretrained=False), tr_dl, te_dl, w, lr=1e-3))
    # 3) transfer — frozen encoder (feature extraction)
    results.append(train_regime("frozen-encoder", make_model(pretrained=True, freeze_encoder=True), tr_dl, te_dl, w, lr=1e-3))
    # 4) transfer — fine-tune all
    results.append(train_regime("fine-tune-all", make_model(pretrained=True), tr_dl, te_dl, w, lr=2e-4))

    # ---- report ----
    print("\n TRANSFER LEARNING (Lecture 08) ", flush=True)
    print(f"Target: {TARGET} (held out from the source model) | {len(Xtr)} adaptation patches", flush=True)
    print(f"{'Regime':28s} {'OA':>7s} {'mIoU':>7s}", flush=True)
    for r in results:
        print(f"{r['name']:28s} {r['oa']:7.3f} {r['miou']:7.3f}", flush=True)
    scratch = next(r for r in results if r["name"] == "scratch")
    ft      = next(r for r in results if r["name"] == "fine-tune-all")
    print(f"\nTransfer gain (fine-tune vs scratch): "
          f"OA {ft['oa']-scratch['oa']:+.3f} | mIoU {ft['miou']-scratch['miou']:+.3f}", flush=True)
    print("Lecture 08: with scarce target labels, reusing pretrained features beats", flush=True)
    print("training from scratch. The optical->SAR domain gap is avoided here because", flush=True)
    print("the source features are themselves SAR, learned on the other regions.", flush=True)
    print(flush=True)

    json.dump(results, open("transfer_results.json", "w"))
    plt.figure(figsize=(10, 5))
    for r in results:
        if r["hist"]:
            plt.plot(range(1, EPOCHS+1), r["hist"], marker="o", ms=3, lw=2, label=r["name"])
    plt.axhline(results[0]["miou"], ls="--", c="grey", label="source zero-shot")
    plt.xlabel("epoch"); plt.ylabel("target-test mIoU"); plt.grid(alpha=.3); plt.legend()
    plt.title(f"Transfer learning on {TARGET} (Lecture 08) — scarce target labels")
    plt.tight_layout(); plt.savefig("transfer_comparison.png", dpi=130); plt.close()
    print("\nSaved: transfer_comparison.png, transfer_results.json", flush=True)


if __name__ == "__main__":
    main()
