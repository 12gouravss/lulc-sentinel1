import os, json, argparse, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import ee, geemap, rasterio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

PROJECT = "lulc-sentinel-513912"
device = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Water","Trees","Grassland","Cropland","Built-up","Bare/Sparse","Wetland"]
PALETTE = ["#1f78b4","#33a02c","#b2df8a","#fdbf6f","#e31a1c","#b15928","#6a3d9a"]
N = len(CLASSES)
CMAP = ListedColormap(PALETTE)
PATCH = 256
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


# models (match training)
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
    def __init__(self, in_dim=7, hidden=64, n_classes=7, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2*hidden, n_classes)
    def forward(self, x):
        h, _ = self.lstm(x); return self.head(h)


# Earth Engine 
def s1_month(aoi, year, month):
    start = ee.Date.fromYMD(year, month, 1); end = start.advance(1, "month")
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(aoi).filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
           .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
           .select(["VV", "VH"]))
    img = col.median().focalMedian(1.5, "circle", "pixels")
    vv = img.select("VV").clamp(-25, 0).unitScale(-25, 0)
    vh = img.select("VH").clamp(-30, -5).unitScale(-30, -5)
    return vv.addBands(vh).rename(["VV", "VH"]).toFloat()


def download(image, aoi, path):
    if os.path.exists(path):          
        return
    geemap.download_ee_image(image.clip(aoi), path, region=aoi, scale=10, crs="EPSG:4326")


def dynamic_world(aoi, year):
    start = ee.Date.fromYMD(year, 1, 1); end = ee.Date.fromYMD(year, 12, 31)
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterBounds(aoi).filterDate(start, end).select("label").mode())
    return dw.remap([0,1,2,3,4,5,6,7,8], [0,1,2,6,3,2,4,5,5]).rename("lulc").toInt()


# inference
def pad_to_patch(a):
    _, h, w = a.shape
    return np.pad(a, ((0,0),(0,(-h)%PATCH),(0,(-w)%PATCH)), mode="reflect"), h, w


@torch.no_grad()
def unet_probs(unet, arr):
    a, h, w = pad_to_patch(np.nan_to_num(arr, nan=0.0))
    _, H, W = a.shape; out = np.zeros((N, H, W), np.float32)
    for iy in range(0, H, PATCH):
        for ix in range(0, W, PATCH):
            t = torch.from_numpy(a[:, iy:iy+PATCH, ix:ix+PATCH]).float().unsqueeze(0).to(device)
            out[:, iy:iy+PATCH, ix:ix+PATCH] = F.softmax(unet(t), 1)[0].cpu().numpy()
    return out[:, :h, :w]


@torch.no_grad()
def lstm_refine(lstm, probs_12):
    T, C, H, W = probs_12.shape
    seq = torch.from_numpy(probs_12.transpose(2, 3, 0, 1).reshape(-1, T, C)).float()
    outs = []
    for b in range(0, len(seq), 200000):
        outs.append(lstm(seq[b:b+200000].to(device)).argmax(-1).cpu())
    return torch.cat(outs).numpy().reshape(H, W, T).transpose(2, 0, 1)


def class_pct(mapHW):
    c = np.bincount(mapHW.ravel(), minlength=N).astype(float)
    return 100 * c / c.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--start-month", type=int, default=1)
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--size-km", type=float, default=5.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--weights", default="unet_s1_lulc.pth")
    ap.add_argument("--lstm", default="lstm_temporal.pth")
    a = ap.parse_args()
    OUT = f"prototype_out_{a.tag}" if a.tag else "prototype_out"
    os.makedirs(OUT, exist_ok=True)

    ee.Initialize(project=PROJECT)
    print(f"EE ready. AOI {a.size_km} km around ({a.lat}, {a.lon}) -> {OUT}/", flush=True)
    aoi = ee.Geometry.Point([a.lon, a.lat]).buffer(a.size_km*1000/2).bounds()

    unet = UNet(base=64).to(device); unet.load_state_dict(torch.load(a.weights, map_location=device)); unet.eval()
    lstm = TemporalLSTM().to(device); lstm.load_state_dict(torch.load(a.lstm, map_location=device)); lstm.eval()

    probs = []; months = []
    for i in range(12):
        m = (a.start_month - 1 + i) % 12 + 1
        yr = a.year + (a.start_month - 1 + i) // 12
        months.append(m)
        tif = f"{OUT}/m{i:02d}_{yr}_{m:02d}.tif"
        print(f"  [{i+1}/12] {MONTH_NAMES[m-1]} {yr} ...", flush=True)
        download(s1_month(aoi, yr, m), aoi, tif)
        with rasterio.open(tif) as r: arr = r.read().astype(np.float32)[:2]
        probs.append(unet_probs(unet, arr))
    probs = np.stack(probs, 0)                        # (12,7,H,W)
    raw_maps = probs.argmax(1)                        # U-Net only
    lstm_maps = lstm_refine(lstm, probs)             # temporally consistent
    conf = probs.max(1).mean(0)                       # A3: mean max-prob per pixel (H,W)
    print("Inference done.", flush=True)

    pct = np.stack([class_pct(lstm_maps[t]) for t in range(12)], 0)   # (12,7)
    total = pct[-1] - pct[0]
    tm = np.zeros((N, N), float)
    for s, d in zip(lstm_maps[0].ravel(), lstm_maps[-1].ravel()): tm[s, d] += 1
    tm_pct = 100 * tm / tm.sum()


    with open(f"{OUT}/class_percentages.csv", "w") as f:
        f.write("month," + ",".join(CLASSES) + "\n")
        for t in range(12):
            f.write(f"{MONTH_NAMES[months[t]-1]}," + ",".join(f"{x:.2f}" for x in pct[t]) + "\n")
    with open(f"{OUT}/total_change_pp.csv", "w") as f:
        f.write("class,month1_%,month12_%,total_change_pp\n")
        for c in range(N):
            f.write(f"{CLASSES[c]},{pct[0,c]:.2f},{pct[-1,c]:.2f},{total[c]:+.2f}\n")

    
    dw_path = f"{OUT}/dynamic_world.tif"; download(dynamic_world(aoi, a.year), aoi, dw_path)
    with rasterio.open(dw_path) as r: dw = np.clip(np.nan_to_num(r.read(1), nan=0), 0, N-1).astype(int)
    dw_pct = class_pct(dw); our_pct = class_pct(lstm_maps[0]); agree = 100 - 0.5*np.abs(dw_pct - our_pct).sum()

    
    #  12 monthly maps
    fig, ax = plt.subplots(3, 4, figsize=(16, 11))
    for t in range(12):
        ax.flat[t].imshow(lstm_maps[t], cmap=CMAP, vmin=0, vmax=N-1)
        ax.flat[t].set_title(f"{MONTH_NAMES[months[t]-1]} {a.year+(a.start_month-1+t)//12}", fontsize=13); ax.flat[t].axis("off")
    fig.legend([plt.Rectangle((0,0),1,1,color=PALETTE[c]) for c in range(N)], CLASSES,
               loc="lower center", ncol=7, fontsize=12)
    fig.suptitle("12-month LULC (temporal-LSTM refined)", fontsize=17)
    plt.tight_layout(rect=[0,0.04,1,0.97]); plt.savefig(f"{OUT}/monthly_maps.png", dpi=130); plt.close()

    #  class % over months
    fig, ax = plt.subplots(figsize=(12, 6))
    for c in range(N): ax.plot(range(1,13), pct[:,c], marker="o", lw=2.5, color=PALETTE[c], label=CLASSES[c])
    ax.set_xticks(range(1,13)); ax.set_xticklabels([MONTH_NAMES[m-1] for m in months])
    ax.set_ylabel("% of area", fontsize=13); ax.set_title("Class area % across 12 months", fontsize=15)
    ax.grid(alpha=.3); ax.legend(ncol=4, fontsize=11)
    plt.tight_layout(); plt.savefig(f"{OUT}/class_percent_over_months.png", dpi=130); plt.close()

    #  transition matrix
    fig, ax = plt.subplots(figsize=(8.5, 7.5)); im = ax.imshow(tm_pct, cmap="inferno")
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("to (month 12)"); ax.set_ylabel("from (month 1)")
    ax.set_title("Class transitions, month 1 -> month 12 (% of area)", fontsize=14)
    for s in range(N):
        for d in range(N):
            if tm_pct[s,d] >= 0.1:
                ax.text(d, s, f"{tm_pct[s,d]:.1f}", ha="center", va="center",
                        color="white" if tm_pct[s,d] < tm_pct.max()*0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046); plt.tight_layout(); plt.savefig(f"{OUT}/transition_matrix.png", dpi=130); plt.close()

    # A2 change-detection map (where change happened, coloured by new class)
    changed = lstm_maps[0] != lstm_maps[-1]
    rgb = np.ones((*changed.shape, 3)) * 0.85          # unchanged = light grey
    import matplotlib.colors as mc
    for c in range(N):
        mask = changed & (lstm_maps[-1] == c)
        rgb[mask] = mc.to_rgb(PALETTE[c])
    pct_changed = 100*changed.mean()
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].imshow(changed, cmap="Reds", vmin=0, vmax=1); ax[0].set_title(f"Where it changed ({pct_changed:.1f}% of area)", fontsize=13); ax[0].axis("off")
    ax[1].imshow(rgb); ax[1].set_title("Change by NEW class (month 12)", fontsize=13); ax[1].axis("off")
    fig.legend([plt.Rectangle((0,0),1,1,color=PALETTE[c]) for c in range(N)], CLASSES, loc="lower center", ncol=7, fontsize=11)
    plt.tight_layout(rect=[0,0.05,1,1]); plt.savefig(f"{OUT}/change_map.png", dpi=130); plt.close()

    #  A3 confidence / uncertainty map
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    im = ax[0].imshow(conf, cmap="viridis", vmin=0.3, vmax=1.0)
    ax[0].set_title("Model confidence (mean max-prob)", fontsize=13); ax[0].axis("off"); fig.colorbar(im, ax=ax[0], fraction=0.046)
    lowmask = conf < 0.5
    ax[1].imshow(lowmask, cmap="Reds", vmin=0, vmax=1)
    ax[1].set_title(f"Low-confidence pixels <0.5 ({100*lowmask.mean():.1f}%)", fontsize=13); ax[1].axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/confidence_map.png", dpi=130); plt.close()

    #  sanity check vs Dynamic World
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    ax[0].imshow(lstm_maps[0], cmap=CMAP, vmin=0, vmax=N-1); ax[0].set_title("Ours (month 1)", fontsize=13); ax[0].axis("off")
    ax[1].imshow(dw, cmap=CMAP, vmin=0, vmax=N-1); ax[1].set_title("Google Dynamic World", fontsize=13); ax[1].axis("off")
    x = np.arange(N); ax[2].bar(x-0.2, our_pct, 0.4, label="Ours", color="#1f78b4"); ax[2].bar(x+0.2, dw_pct, 0.4, label="Dynamic World", color="#e31a1c")
    ax[2].set_xticks(x); ax[2].set_xticklabels(CLASSES, rotation=45, ha="right")
    ax[2].set_ylabel("% of area"); ax[2].set_title(f"Area-share agreement: {agree:.1f}%", fontsize=13); ax[2].legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/sanity_check_dynamic_world.png", dpi=130); plt.close()


    print("\n PROTOTYPE SUMMARY ", flush=True)
    print(f"Location ({a.lat}, {a.lon}) | {a.size_km} km | start {MONTH_NAMES[a.start_month-1]} {a.year}", flush=True)
    print("\nTotal first->last change (percentage points):", flush=True)
    for c in range(N):
        print(f"  {CLASSES[c]:12s} {pct[0,c]:6.2f}% -> {pct[-1,c]:6.2f}%   ({total[c]:+.2f} pp)", flush=True)
    print("\nTop transitions month1 -> month12:", flush=True)
    for v, s, d in sorted([(tm_pct[s,d], s, d) for s in range(N) for d in range(N) if s != d], reverse=True)[:6]:
        print(f"  {CLASSES[s]:12s} -> {CLASSES[d]:12s} {v:5.2f}% of area", flush=True)
    print(f"\nChanged area: {pct_changed:.1f}% | mean confidence: {conf.mean():.3f} | low-conf <0.5: {100*lowmask.mean():.1f}%", flush=True)
    print(f"Dynamic World sanity check: {agree:.1f}% area-share agreement", flush=True)
    print(flush=True)
    json.dump({"lat": a.lat, "lon": a.lon, "pct": pct.tolist(), "total_change_pp": total.tolist(),
               "pct_changed": float(pct_changed), "mean_confidence": float(conf.mean()), "dw_agreement": agree},
              open(f"{OUT}/summary.json", "w"))
    print(f"\nAll outputs saved in {OUT}/", flush=True)


if __name__ == "__main__":
    main()
