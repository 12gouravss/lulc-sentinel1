import re, torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train import UNet, load_clean, SARDataset, evaluate, device, N, CLASSES

Xte, Yte = load_clean("test.npz")
m = UNet(in_ch=2, n_classes=N, base=64).to(device)
m.load_state_dict(torch.load("unet_s1_lulc.pth", map_location=device)); m.eval()
oa, miou, iou, cm = evaluate(m, DataLoader(SARDataset(Xte, Yte, [0,1]), batch_size=16))
print(f"\n=== BASELINE (VV+VH) HELD-OUT TEST ===  OA {oa:.3f} | mIoU {miou:.3f}")
for c in range(N): print(f"  {CLASSES[c]:12s} IoU {iou[c]:.3f}")

cmn = cm / cm.sum(1, keepdims=True).clip(1)
plt.figure(figsize=(7,6)); plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
plt.xticks(range(N), CLASSES, rotation=45, ha="right"); plt.yticks(range(N), CLASSES)
for i in range(N):
    for j in range(N):
        plt.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=8,
                 color="white" if cmn[i,j] > 0.5 else "black")
plt.xlabel("predicted"); plt.ylabel("true")
plt.title(f"Baseline confusion matrix (OA {oa:.3f}, mIoU {miou:.3f})")
plt.colorbar(fraction=0.046); plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=130); plt.close()

ep, oas, mis = [], [], []
for ln in open("train.log"):
    g = re.search(r"ep (\d+) .*OA ([\d.]+) \| mIoU ([\d.]+)", ln)
    if g: ep.append(int(g.group(1))); oas.append(float(g.group(2))); mis.append(float(g.group(3)))
if ep:
    plt.figure(figsize=(10,5))
    plt.plot(ep, oas, "-o", ms=3, label="val OA"); plt.plot(ep, mis, "-o", ms=3, label="val mIoU")
    plt.xlabel("epoch"); plt.ylabel("score"); plt.grid(alpha=.3); plt.legend()
    plt.title("Baseline U-Net training curves"); plt.tight_layout()
    plt.savefig("training_curves.png", dpi=130); plt.close()
print("confusion_matrix.png, training_curves.png")
