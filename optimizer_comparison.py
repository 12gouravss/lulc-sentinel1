import numpy as np, torch, json
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

device="cuda" if torch.cuda.is_available() else "cpu"
CLASSES=["Water","Trees","Grassland","Cropland","Built-up","Bare/Sparse","Wetland"]; N=len(CLASSES)
EPOCHS=18; BATCH=16; BASE=64

def load_clean(path):
    d=np.load(path); X=d["X"].astype(np.float32)
    Y=np.clip(np.round(np.nan_to_num(d["Y"].astype(np.float32),nan=0.0)),0,N-1).astype(np.uint8)
    return X,Y

class DS(Dataset):
    def __init__(s,X,Y,aug=False): s.X,s.Y,s.aug=X,Y,aug
    def __len__(s): return len(s.X)
    def __getitem__(s,i):
        x,y=s.X[i].copy(),s.Y[i].copy()
        if s.aug:
            if np.random.rand()<.5: x,y=np.fliplr(x).copy(),np.fliplr(y).copy()
            if np.random.rand()<.5: x,y=np.flipud(x).copy(),np.flipud(y).copy()
            k=np.random.randint(4); x,y=np.rot90(x,k).copy(),np.rot90(y,k).copy()
        return torch.from_numpy(x).permute(2,0,1).float(),torch.from_numpy(y).long()

class DoubleConv(nn.Module):
    def __init__(s,a,b):
        super().__init__(); s.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(True),
                                                nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(True))
    def forward(s,x): return s.net(x)
class UNet(nn.Module):
    def __init__(s,in_ch=2,n=N,base=BASE):
        super().__init__()
        s.d1=DoubleConv(in_ch,base);s.d2=DoubleConv(base,base*2);s.d3=DoubleConv(base*2,base*4);s.d4=DoubleConv(base*4,base*8)
        s.pool=nn.MaxPool2d(2);s.bott=DoubleConv(base*8,base*16)
        s.up4=nn.ConvTranspose2d(base*16,base*8,2,2);s.u4=DoubleConv(base*16,base*8)
        s.up3=nn.ConvTranspose2d(base*8,base*4,2,2);s.u3=DoubleConv(base*8,base*4)
        s.up2=nn.ConvTranspose2d(base*4,base*2,2,2);s.u2=DoubleConv(base*4,base*2)
        s.up1=nn.ConvTranspose2d(base*2,base,2,2);s.u1=DoubleConv(base*2,base)
        s.out=nn.Conv2d(base,n,1)
    def forward(s,x):
        c1=s.d1(x);c2=s.d2(s.pool(c1));c3=s.d3(s.pool(c2));c4=s.d4(s.pool(c3));b=s.bott(s.pool(c4))
        x=s.u4(torch.cat([s.up4(b),c4],1));x=s.u3(torch.cat([s.up3(x),c3],1))
        x=s.u2(torch.cat([s.up2(x),c2],1));x=s.u1(torch.cat([s.up1(x),c1],1));return s.out(x)

def dice(lo,t,eps=1e-6):
    p=F.softmax(lo,1); oh=F.one_hot(t,N).permute(0,3,1,2).float()
    i=(p*oh).sum((0,2,3)); u=p.sum((0,2,3))+oh.sum((0,2,3)); return 1-((2*i+eps)/(u+eps)).mean()
def cw(Y):
    c=np.bincount(Y.ravel(),minlength=N).astype(float); return torch.tensor(c.sum()/(N*np.maximum(c,1)),dtype=torch.float32)
@torch.no_grad()
def evaluate(model,dl):
    model.eval(); inter=np.zeros(N); union=np.zeros(N); cor=tot=0
    for x,y in dl:
        x=x.to(device); y=y.to(device); p=model(x).argmax(1); cor+=(p==y).sum().item(); tot+=y.numel()
        for c in range(N):
            pm=(p==c); g=(y==c); inter[c]+=(pm&g).sum().item(); union[c]+=(pm|g).sum().item()
    return cor/tot,(inter/np.maximum(union,1)).mean()

def make_opt(name,params):
    if name=="SGD+momentum": return torch.optim.SGD(params,lr=0.01,momentum=0.9)
    if name=="Adam":         return torch.optim.Adam(params,lr=1e-3)
    if name=="AdamW":        return torch.optim.AdamW(params,lr=1e-3,weight_decay=1e-2)

def train_one(name,tr,va,te,w):
    torch.manual_seed(0); np.random.seed(0)
    model=UNet().to(device); opt=make_opt(name,model.parameters())
    ce=nn.CrossEntropyLoss(weight=w); crit=lambda lo,t: ce(lo,t)+dice(lo,t)
    hist=[]
    for ep in range(1,EPOCHS+1):
        model.train()
        for x,y in tr:
            x=x.to(device); y=y.to(device); opt.zero_grad(); crit(model(x),y).backward(); opt.step()
        oa,miou=evaluate(model,va); hist.append((oa,miou))
        print(f"  [{name}] ep {ep:02d} | val OA {oa:.3f} | mIoU {miou:.3f}",flush=True)
    toa,tmiou=evaluate(model,te)
    print(f"  [{name}] TEST OA {toa:.3f} | mIoU {tmiou:.3f}",flush=True)
    return {"name":name,"hist":hist,"test_oa":toa,"test_miou":tmiou}

def main():
    Xtr,Ytr=load_clean("train.npz"); Xte,Yte=load_clean("test.npz")
    rng=np.random.RandomState(0); idx=rng.permutation(len(Xtr)); nv=int(.15*len(idx)); vi,ti=idx[:nv],idx[nv:]
    tr=DataLoader(DS(Xtr[ti],Ytr[ti],aug=True),batch_size=BATCH,shuffle=True)
    va=DataLoader(DS(Xtr[vi],Ytr[vi]),batch_size=BATCH); te=DataLoader(DS(Xte,Yte),batch_size=BATCH)
    w=cw(Ytr[ti]).to(device)
    print(f"Lecture-05 optimizer comparison | {EPOCHS} epochs each | device {device}",flush=True)

    results=[train_one(n,tr,va,te,w) for n in ["SGD+momentum","Adam","AdamW"]]

    print("\n OPTIMIZER COMPARISON (test set) ",flush=True)
    print(f"{'Optimizer':16s} {'Test OA':>9s} {'Test mIoU':>10s}",flush=True)
    for r in results: print(f"{r['name']:16s} {r['test_oa']:9.3f} {r['test_miou']:10.3f}",flush=True)
    best=max(results,key=lambda r:r['test_miou'])
    print(f"\nBest generalization: {best['name']} (test mIoU {best['test_miou']:.3f})",flush=True)
    print("Consistent with lecture 05: the optimizer is an empirical choice — AdamW's",flush=True)
    print("decoupled weight decay often generalizes best, but it must be verified, not assumed.",flush=True)
    print(flush=True)

    json.dump(results,open("optimizer_comparison.json","w"))
    plt.figure(figsize=(11,6))
    for r in results: plt.plot(range(1,EPOCHS+1),[h[1] for h in r["hist"]],marker="o",lw=2,label=r["name"])
    plt.xlabel("epoch"); plt.ylabel("validation mIoU"); plt.grid(alpha=.3); plt.legend()
    plt.title("Optimizer comparison (lecture 05) — validation mIoU"); plt.tight_layout()
    plt.savefig("optimizer_comparison.png",dpi=130); plt.close()
    print("\nSaved: optimizer_comparison.png, optimizer_comparison.json",flush=True)

if __name__=="__main__":
    main()
