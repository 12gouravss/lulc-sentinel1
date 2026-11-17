import os, glob, re, json, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import rasterio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Water","Trees","Grassland","Cropland","Built-up","Bare/Sparse","Wetland"]
N = len(CLASSES)
PATCH = 256
MONTHS = list(range(1, 13))
PIX_PER_PATCH = 250 
MAX_SEQ = 200_000 


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



class TemporalLSTM(nn.Module):
    """Bidirectional LSTM over a 12-step sequence of 7-dim class-probability vectors."""
    def __init__(self, in_dim=7, hidden=64, n_classes=7, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2*hidden, n_classes)
    def forward(self, x):            # x: (B, 12, 7)
        h, _ = self.lstm(x)          # (B, 12, 2*hidden)
        return self.head(h)          # (B, 12, 7)



@torch.no_grad()
def build_sequences(unet):
    region_files = sorted(glob.glob("tiles/*_2021_01.tif"))
    regions = [re.match(r"tiles/(.+)_2021_01\.tif$", f).group(1) for f in region_files]
    print(f"Regions found: {regions}", flush=True)

    seqs, labels = [], []
    rng = np.random.RandomState(0)
    for reg in regions:
        # read all 12 months (bands: 1=VV, 2=VH, 3=label)
        months = []
        label = None
        ok = True
        for m in MONTHS:
            path = f"tiles/{reg}_2021_{m:02d}.tif"
            if not os.path.exists(path): ok = False; break
            with rasterio.open(path) as r:
                a = r.read().astype(np.float32)      # (3, H, W)
            months.append(a[:2])                     # VV, VH
            if m == 1: label = a[2]
        if not ok:
            print(f"  {reg}: missing a month, skipping", flush=True); continue

        H = min(x.shape[1] for x in months); W = min(x.shape[2] for x in months)
        H -= H % PATCH; W -= W % PATCH
        months = [x[:, :H, :W] for x in months]; label = label[:H, :W]
        stack = np.stack(months, 0)                  # (12, 2, H, W)
        label = np.clip(np.nan_to_num(label, nan=0.0), 0, N-1).astype(np.int64)

        nY, nX = H // PATCH, W // PATCH
        for iy in range(nY):
            for ix in range(nX):
                ys, xs = iy*PATCH, ix*PATCH
                patch = stack[:, :, ys:ys+PATCH, xs:xs+PATCH]          # (12,2,256,256)
                patch = np.nan_to_num(patch, nan=0.0)
                xb = torch.from_numpy(patch).float().to(device)        # 12 months as a batch
                prob = F.softmax(unet(xb), dim=1).cpu().numpy()        # (12,7,256,256)
                lp = label[ys:ys+PATCH, xs:xs+PATCH]                   # (256,256)

                
                flat = PATCH*PATCH
                idx = rng.choice(flat, size=min(PIX_PER_PATCH, flat), replace=False)
                pr = prob.reshape(12, N, flat)[:, :, idx]              # (12,7,K)
                pr = np.transpose(pr, (2, 0, 1))                       # (K,12,7)
                yl = lp.reshape(flat)[idx]                             # (K,)
                seqs.append(pr.astype(np.float32)); labels.append(yl)
        print(f"  {reg}: sequences so far {sum(len(s) for s in seqs):,}", flush=True)
        if sum(len(s) for s in seqs) >= MAX_SEQ: break

    X = np.concatenate(seqs, 0)[:MAX_SEQ]
    y = np.concatenate(labels, 0)[:MAX_SEQ]
    print(f"Total sequences: X {X.shape} y {y.shape}", flush=True)
    return X, y



def avg_switches(seq_argmax):
    """Mean number of class changes across the 12 months, per pixel."""
    return float((seq_argmax[:, 1:] != seq_argmax[:, :-1]).sum(1).mean())


def main():
    if not os.path.exists("unet_s1_lulc.pth"):
        print("ERROR: unet_s1_lulc.pth not found in the Vault."); return
    unet = UNet(in_ch=2, n_classes=N, base=64).to(device)
    unet.load_state_dict(torch.load("unet_s1_lulc.pth", map_location=device)); unet.eval()
    print("Loaded U-Net. Building temporal sequences from the 12-month tiles...", flush=True)

    X, y = build_sequences(unet)
    Xt = torch.from_numpy(X); yt = torch.from_numpy(y)

    # split
    n = len(X); idx = np.random.RandomState(0).permutation(n); nval = int(0.15*n)
    vi, ti = idx[:nval], idx[nval:]
    Xtr, ytr = Xt[ti].to(device), yt[ti].to(device)
    Xva, yva = Xt[vi].to(device), yt[vi].to(device)

    lstm = TemporalLSTM(in_dim=N, hidden=64, n_classes=N).to(device)
    opt = torch.optim.Adam(lstm.parameters(), lr=1e-3)
    print(f"Training LSTM | params {sum(p.numel() for p in lstm.parameters())/1e3:.0f}k | "
          f"train {len(ti):,} val {len(vi):,}", flush=True)

    BATCH = 4096
    for ep in range(1, 31):
        lstm.train(); perm = torch.randperm(len(Xtr), device=device); tl = 0; nb = 0
        for b in range(0, len(Xtr), BATCH):
            bi = perm[b:b+BATCH]; xb = Xtr[bi]; yb = yva if False else ytr[bi]
            opt.zero_grad()
            out = lstm(xb)                                   # (B,12,7)
            loss = F.cross_entropy(out.reshape(-1, N), yb.repeat_interleave(12))
            loss.backward(); opt.step(); tl += loss.item(); nb += 1
        # val per-timestep accuracy
        lstm.eval()
        with torch.no_grad():
            vout = lstm(Xva).argmax(-1)                      # (B,12)
            vacc = (vout == yva[:, None]).float().mean().item()
        print(f"ep {ep:02d} | loss {tl/nb:.4f} | val per-month acc {vacc:.3f}", flush=True)

    torch.save(lstm.state_dict(), "lstm_temporal.pth")

    # phantom-change analysis on the validation set
    with torch.no_grad():
        raw_argmax = Xva.argmax(-1).cpu().numpy()            # U-Net per-month argmax
        lstm_argmax = lstm(Xva).argmax(-1).cpu().numpy()     # LSTM-smoothed
    raw_sw = avg_switches(raw_argmax); lstm_sw = avg_switches(lstm_argmax)
    red = 100*(1 - lstm_sw/max(raw_sw, 1e-9))

    # accuracy vs stable annual label (consistency shouldn't hurt accuracy)
    yv = yva.cpu().numpy()
    raw_acc = float((raw_argmax == yv[:, None]).mean())
    lstm_acc = float((lstm_argmax == yv[:, None]).mean())

    print("\n TEMPORAL LSTM RESULTS ", flush=True)
    print(f"Avg class switches / pixel over 12 months:", flush=True)
    print(f"   raw U-Net (per-month argmax): {raw_sw:.3f}", flush=True)
    print(f"   LSTM-smoothed              : {lstm_sw:.3f}", flush=True)
    print(f"   PHANTOM CHANGE REDUCED BY  : {red:.1f}%", flush=True)
    print(f"Per-month accuracy vs annual label: raw {raw_acc:.3f} -> LSTM {lstm_acc:.3f}", flush=True)
    print( flush=True)

    json.dump({"raw_switches": raw_sw, "lstm_switches": lstm_sw, "reduction_pct": red,
               "raw_acc": raw_acc, "lstm_acc": lstm_acc, "val_month_acc": vacc},
              open("lstm_results.json", "w"))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["raw U-Net", "LSTM-smoothed"], [raw_sw, lstm_sw], color=["#e31a1c", "#1f78b4"])
    ax.set_ylabel("avg class switches per pixel (12 months)")
    ax.set_title(f"Temporal consistency — phantom change reduced {red:.0f}%")
    for i, v in enumerate([raw_sw, lstm_sw]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=12)
    plt.tight_layout(); plt.savefig("lstm_switch_reduction.png", dpi=120); plt.close()
    print("\nDONE -> lstm_temporal.pth, lstm_results.json, lstm_switch_reduction.png", flush=True)


if __name__ == "__main__":
    main()
