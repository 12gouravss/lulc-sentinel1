import os, glob, re, json, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Water","Trees","Grassland","Cropland","Built-up","Bare/Sparse","Wetland"]
N = len(CLASSES); PATCH = 256; IN_CH = 10
TRAIN_REGIONS = ["po_valley","rhine_neckar","flevoland","andalusia","danube","saxony","alps","greece_scrub","camargue"]
TEST_REGIONS  = ["beauce","catalonia","provence"]
BASELINE = {"oa": 0.757, "miou": 0.347}      # single-month model, for comparison


# build temporal-feature patches
def region_features(reg):
    """12 monthly tiles -> (10,H,W) temporal features + (H,W) annual label."""
    months, label = [], None
    for m in range(1, 13):
        p = f"tiles/{reg}_2021_{m:02d}.tif"
        if not os.path.exists(p): return None, None
        with rasterio.open(p) as r: a = r.read().astype(np.float32)
        months.append(a[:2]); label = a[2] if m == 1 else label
    s = np.nan_to_num(np.stack(months, 0), nan=0.0)      # (12,2,H,W)
    vv, vh = s[:,0], s[:,1]; d = vv - vh
    feats = np.stack([vv.mean(0), vv.std(0), vv.min(0), vv.max(0),
                      vh.mean(0), vh.std(0), vh.min(0), vh.max(0),
                      d.mean(0),  d.std(0)], 0).astype(np.float32)   # (10,H,W)
    label = np.clip(np.nan_to_num(label, nan=0.0), 0, N-1).astype(np.uint8)
    return feats, label

def build(regions):
    X, Y = [], []
    for reg in regions:
        f, y = region_features(reg)
        if f is None: print("  skip", reg, flush=True); continue
        H, W = y.shape; H -= H % PATCH; W -= W % PATCH
        for i in range(0, H, PATCH):
            for j in range(0, W, PATCH):
                xp = f[:, i:i+PATCH, j:j+PATCH]; yp = y[i:i+PATCH, j:j+PATCH]
                X.append(xp.transpose(1,2,0)); Y.append(yp)
        print(f"  {reg}: {len(X)} patches", flush=True)
    return np.array(X, np.float32), np.array(Y, np.uint8)


class DS(Dataset):
    def __init__(s, X, Y, aug=False): s.X, s.Y, s.aug = X, Y, aug
    def __len__(s): return len(s.X)
    def __getitem__(s, i):
        x, y = s.X[i].copy(), s.Y[i].copy()
        if s.aug:
            if np.random.rand()<.5: x,y = np.fliplr(x).copy(), np.fliplr(y).copy()
            if np.random.rand()<.5: x,y = np.flipud(x).copy(), np.flipud(y).copy()
            k = np.random.randint(4); x,y = np.rot90(x,k).copy(), np.rot90(y,k).copy()
        return torch.from_numpy(x).permute(2,0,1).float(), torch.from_numpy(y).long()


# model (U-Net, 10->7)
class DoubleConv(nn.Module):
    def __init__(s,a,b):
        super().__init__(); s.n=nn.Sequential(
            nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(True),
            nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(True))
    def forward(s,x): return s.n(x)
class UNet(nn.Module):
    def __init__(s,in_ch=IN_CH,n=N,base=64):
        super().__init__()
        s.d1=DoubleConv(in_ch,base);s.d2=DoubleConv(base,base*2);s.d3=DoubleConv(base*2,base*4);s.d4=DoubleConv(base*4,base*8)
        s.p=nn.MaxPool2d(2);s.b=DoubleConv(base*8,base*16)
        s.u4=nn.ConvTranspose2d(base*16,base*8,2,2);s.c4=DoubleConv(base*16,base*8)
        s.u3=nn.ConvTranspose2d(base*8,base*4,2,2);s.c3=DoubleConv(base*8,base*4)
        s.u2=nn.ConvTranspose2d(base*4,base*2,2,2);s.c2=DoubleConv(base*4,base*2)
        s.u1=nn.ConvTranspose2d(base*2,base,2,2);s.c1=DoubleConv(base*2,base)
        s.o=nn.Conv2d(base,n,1)
    def forward(s,x):
        a=s.d1(x);b=s.d2(s.p(a));c=s.d3(s.p(b));d=s.d4(s.p(c));e=s.b(s.p(d))
        x=s.c4(torch.cat([s.u4(e),d],1));x=s.c3(torch.cat([s.u3(x),c],1))
        x=s.c2(torch.cat([s.u2(x),b],1));x=s.c1(torch.cat([s.u1(x),a],1));return s.o(x)


# losses / metrics (#3, #5)
def dice(logits,t,eps=1e-6):
    p=F.softmax(logits,1); oh=F.one_hot(t,N).permute(0,3,1,2).float()
    i=(p*oh).sum((0,2,3)); u=p.sum((0,2,3))+oh.sum((0,2,3)); return 1-((2*i+eps)/(u+eps)).mean()
def focal(logits,t,w,g=2.0):
    logp=F.log_softmax(logits,1); ce=F.nll_loss(logp,t,weight=w,reduction="none")
    pt=logp.exp().gather(1,t.unsqueeze(1)).squeeze(1); return ((1-pt)**g*ce).mean()
def cweights(Y):
    c=np.bincount(Y.ravel(),minlength=N).astype(float); return torch.tensor(c.sum()/(N*np.maximum(c,1)),dtype=torch.float32)

@torch.no_grad()
def tta(model,x):                                   # #5 test-time augmentation
    o =F.softmax(model(x),1)
    o+=torch.flip(F.softmax(model(torch.flip(x,[3])),1),[3])
    o+=torch.flip(F.softmax(model(torch.flip(x,[2])),1),[2])
    return o/3
@torch.no_grad()
def evaluate(model,dl,use_tta=False):
    model.eval(); inter=np.zeros(N); union=np.zeros(N); cor=tot=0; cm=np.zeros((N,N),np.int64)
    for x,y in dl:
        x=x.to(device); y=y.to(device)
        pred=(tta(model,x) if use_tta else F.softmax(model(x),1)).argmax(1)
        cor+=(pred==y).sum().item(); tot+=y.numel()
        for c in range(N):
            p=(pred==c); g=(y==c); inter[c]+=(p&g).sum().item(); union[c]+=(p|g).sum().item()
        t=y.cpu().numpy().ravel(); pp=pred.cpu().numpy().ravel(); cm+=np.bincount(N*t+pp,minlength=N*N).reshape(N,N)
    iou=inter/np.maximum(union,1); return cor/tot, iou.mean(), iou, cm


def main():
    print("Building multi-temporal features from the 12-month tiles…", flush=True)
    Xtr,Ytr=build(TRAIN_REGIONS); Xte,Yte=build(TEST_REGIONS)
    print(f"train {Xtr.shape} test {Xte.shape}", flush=True)
    rng=np.random.RandomState(0); idx=rng.permutation(len(Xtr)); nv=int(.15*len(idx))
    vi,ti=idx[:nv],idx[nv:]
    tr=DataLoader(DS(Xtr[ti],Ytr[ti],aug=True),batch_size=16,shuffle=True)
    va=DataLoader(DS(Xtr[vi],Ytr[vi]),batch_size=16)
    te=DataLoader(DS(Xte,Yte),batch_size=16)

    model=UNet().to(device)
    w=cweights(Ytr[ti]).to(device)
    crit=lambda lo,t: focal(lo,t,w)+dice(lo,t)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=0.5,patience=5)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M | in_ch {IN_CH} | "
          f"train {len(ti)} val {len(vi)} test {len(Xte)}", flush=True)

    best=0; wait=0; hist={"oa":[],"miou":[]}
    for ep in range(1,81):
        model.train()
        for x,y in tr:
            x=x.to(device); y=y.to(device); opt.zero_grad(); loss=crit(model(x),y); loss.backward(); opt.step()
        oa,miou,_,_=evaluate(model,va); sch.step(miou); hist["oa"].append(oa); hist["miou"].append(miou)
        flag=""
        if miou>best: best=miou; wait=0; torch.save(model.state_dict(),"unet_multitemporal.pth"); flag=" *"
        else: wait+=1
        print(f"ep {ep:02d} | val OA {oa:.3f} | mIoU {miou:.3f} | lr {opt.param_groups[0]['lr']:.1e}{flag}", flush=True)
        if wait>=12: print("early stop",flush=True); break

    model.load_state_dict(torch.load("unet_multitemporal.pth",map_location=device))
    oa,miou,per,cm=evaluate(model,te,use_tta=True)       # TTA on the test set
    print(f"\n=== MULTI-TEMPORAL TEST (with TTA) ===  OA {oa:.3f} | mIoU {miou:.3f}", flush=True)
    print(f"{'Class':12s} {'IoU':>6s} {'F1':>6s}", flush=True)
    per_out={}
    for c in range(N):
        tp=cm[c,c]; fp=cm[:,c].sum()-tp; fn=cm[c,:].sum()-tp
        pr=tp/max(tp+fp,1); re=tp/max(tp+fn,1); f1=2*pr*re/max(pr+re,1e-9)
        per_out[CLASSES[c]]={"iou":float(per[c]),"f1":float(f1)}
        print(f"{CLASSES[c]:12s} {per[c]:6.3f} {f1:6.3f}", flush=True)

    print("\n ACCURACY COMPARISON ", flush=True)
    print(f"{'Model':26s} {'OA':>7s} {'mIoU':>7s}", flush=True)
    print(f"{'Single-month U-Net':26s} {BASELINE['oa']:7.3f} {BASELINE['miou']:7.3f}", flush=True)
    print(f"{'Multi-temporal U-Net+TTA':26s} {oa:7.3f} {miou:7.3f}", flush=True)
    print(f"{'improvement':26s} {oa-BASELINE['oa']:+7.3f} {miou-BASELINE['miou']:+7.3f}", flush=True)
    print( flush=True)

    json.dump({"test_oa":float(oa),"test_miou":float(miou),"per_class":per_out,
               "baseline":BASELINE,"history":hist}, open("results_multitemporal.json","w"))
    ep=range(1,len(hist["oa"])+1)
    plt.figure(figsize=(10,5)); plt.plot(ep,hist["oa"],label="val OA"); plt.plot(ep,hist["miou"],label="val mIoU")
    plt.axhline(BASELINE["oa"],ls="--",c="grey",label="single-month OA")
    plt.xlabel("epoch"); plt.ylim(0,1); plt.legend(); plt.grid(alpha=.3); plt.title("Multi-temporal model")
    plt.tight_layout(); plt.savefig("multitemporal_curves.png",dpi=120); plt.close()
    print("\nDONE -> unet_multitemporal.pth, results_multitemporal.json, multitemporal_curves.png", flush=True)


if __name__ == "__main__":
    main()
