import os, argparse, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import ee, geemap, rasterio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PROJECT="lulc-sentinel-513912"; device="cuda" if torch.cuda.is_available() else "cpu"
CLASSES=["Water","Trees","Grassland","Cropland","Built-up","Bare/Sparse","Wetland"]
PALETTE=["#1f78b4","#33a02c","#b2df8a","#fdbf6f","#e31a1c","#b15928","#6a3d9a"]
N=len(CLASSES); PATCH=256; SCALE=50; OUT="forecast_out"


class DCb(nn.Module):
    def __init__(s,a,b):
        super().__init__(); s.n=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(True),
                                              nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(True))
    def forward(s,x): return s.n(x)
class UNetMT(nn.Module):
    def __init__(s,in_ch=10,n=N,base=64):
        super().__init__()
        s.d1=DCb(in_ch,base);s.d2=DCb(base,base*2);s.d3=DCb(base*2,base*4);s.d4=DCb(base*4,base*8)
        s.p=nn.MaxPool2d(2);s.b=DCb(base*8,base*16)
        s.u4=nn.ConvTranspose2d(base*16,base*8,2,2);s.c4=DCb(base*16,base*8)
        s.u3=nn.ConvTranspose2d(base*8,base*4,2,2);s.c3=DCb(base*8,base*4)
        s.u2=nn.ConvTranspose2d(base*4,base*2,2,2);s.c2=DCb(base*4,base*2)
        s.u1=nn.ConvTranspose2d(base*2,base,2,2);s.c1=DCb(base*2,base)
        s.o=nn.Conv2d(base,n,1)
    def forward(s,x):
        a=s.d1(x);b=s.d2(s.p(a));c=s.d3(s.p(b));d=s.d4(s.p(c));e=s.b(s.p(d))
        x=s.c4(torch.cat([s.u4(e),d],1));x=s.c3(torch.cat([s.u3(x),c],1))
        x=s.c2(torch.cat([s.u2(x),b],1));x=s.c1(torch.cat([s.u1(x),a],1));return s.o(x)

# small forecasting LSTM 
class ForecastLSTM(nn.Module):
    def __init__(s,dim=N,hidden=24):
        super().__init__(); s.lstm=nn.LSTM(dim,hidden,batch_first=True); s.head=nn.Linear(hidden,dim)
    def forward(s,x): h,_=s.lstm(x); return s.head(h[:,-1])     # predict next vector from the sequence


def s1_month(aoi,year,month):
    st=ee.Date.fromYMD(year,month,1); en=st.advance(1,"month")
    col=(ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(st,en)
         .filter(ee.Filter.eq("instrumentMode","IW"))
         .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV"))
         .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VH"))
         .filter(ee.Filter.eq("orbitProperties_pass","DESCENDING")).select(["VV","VH"]))
    img=col.median().focalMedian(1.5,"circle","pixels")
    vv=img.select("VV").clamp(-25,0).unitScale(-25,0); vh=img.select("VH").clamp(-30,-5).unitScale(-30,-5)
    return vv.addBands(vh).rename(["VV","VH"]).toFloat()
def s1_stack(aoi,year):
    return ee.Image.cat([s1_month(aoi,year,m).rename([f"VV{m:02d}",f"VH{m:02d}"]) for m in range(1,13)]).toFloat()
def download(image,aoi,path,scale=SCALE):
    if not os.path.exists(path): geemap.download_ee_image(image.clip(aoi),path,region=aoi,scale=scale,crs="EPSG:4326")
def features_from_stack(raw):
    vv=raw[0::2]; vh=raw[1::2]; d=vv-vh
    return np.stack([vv.mean(0),vv.std(0),vv.min(0),vv.max(0),vh.mean(0),vh.std(0),vh.min(0),vh.max(0),d.mean(0),d.std(0)],0).astype(np.float32)
@torch.no_grad()
def classify(model,feats):
    a=np.nan_to_num(feats,nan=0.0); _,h,w=a.shape
    a=np.pad(a,((0,0),(0,(-h)%PATCH),(0,(-w)%PATCH)),mode="reflect"); _,H,W=a.shape; out=np.zeros((H,W),np.int64)
    for iy in range(0,H,PATCH):
        for ix in range(0,W,PATCH):
            t=torch.from_numpy(a[:,iy:iy+PATCH,ix:ix+PATCH]).float().unsqueeze(0).to(device)
            out[iy:iy+PATCH,ix:ix+PATCH]=model(t).argmax(1)[0].cpu().numpy()
    return out[:h,:w]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lat",type=float,required=True); ap.add_argument("--lon",type=float,required=True)
    ap.add_argument("--years",type=int,nargs="+",required=True)
    ap.add_argument("--predict",type=int,default=2); ap.add_argument("--size-km",type=float,default=20)
    ap.add_argument("--weights",default="unet_multitemporal.pth")
    a=ap.parse_args(); os.makedirs(OUT,exist_ok=True); years=sorted(a.years)
    ee.Initialize(project=PROJECT)
    aoi=ee.Geometry.Point([a.lon,a.lat]).buffer(a.size_km*1000/2).bounds()
    model=UNetMT(10).to(device); model.load_state_dict(torch.load(a.weights,map_location=device)); model.eval()

    #  one land-cover map per year  -> per-class % time series
    maps=[]
    for y in years:
        tif=f"{OUT}/stack_{y}_s{SCALE}.tif"
        print(f"  building {y} annual map…",flush=True)
        download(s1_stack(aoi,y),aoi,tif);
        with rasterio.open(tif) as r: raw=r.read().astype(np.float32)
        maps.append(classify(model,features_from_stack(raw)))
    H=min(m.shape[0] for m in maps); W=min(m.shape[1] for m in maps); maps=[m[:H,:W] for m in maps]
    series=np.stack([np.bincount(m.ravel(),minlength=N)/m.size for m in maps],0)   # (T,7) fractions
    T=len(years)

    # MARKOV baseline — average annual transition matrix, projected forward
    P=np.zeros((N,N))
    for t in range(T-1):
        for i in range(N):
            mask=maps[t]==i
            if mask.sum()>0: P[i]+=np.bincount(maps[t+1][mask],minlength=N)/mask.sum()
    P/=max(T-1,1); P=P/P.sum(1,keepdims=True).clip(1e-9)
    d=series[-1].copy(); markov=[]
    for _ in range(a.predict): d=d@P; markov.append(d.copy())
    markov=np.array(markov)

    # LSTM forecast — lectures 10-11 (noise-augmented so the tiny series is trainable)
    W_=min(3,T-1); torch.manual_seed(0); np.random.seed(0)
    Xtr,Ytr=[],[]
    for _ in range(80):
        s=np.clip(series+np.random.normal(0,0.01,series.shape),0,1)
        for t in range(W_,T): Xtr.append(s[t-W_:t]); Ytr.append(s[t])
    Xtr=torch.tensor(np.array(Xtr)).float(); Ytr=torch.tensor(np.array(Ytr)).float()
    net=ForecastLSTM().to(device); opt=torch.optim.Adam(net.parameters(),lr=5e-3)
    for ep in range(400):
        opt.zero_grad(); loss=F.mse_loss(net(Xtr.to(device)),Ytr.to(device)); loss.backward(); opt.step()
    seq=list(series.copy()); lstm=[]
    with torch.no_grad():
        for _ in range(a.predict):
            inp=torch.tensor(np.array(seq[-W_:])[None]).float().to(device)
            nxt=net(inp).cpu().numpy()[0]; nxt=np.clip(nxt,0,None); nxt/=nxt.sum()
            lstm.append(nxt); seq.append(nxt)
    lstm=np.array(lstm)
    fut=[years[-1]+k+1 for k in range(a.predict)]

    
    print("\n LAND-COVER FORECAST ",flush=True)
    print(f"Location ({a.lat},{a.lon}) | observed {years[0]}–{years[-1]} | predicting {fut}",flush=True)
    print(f"\n{'Class':12s} {years[-1]:>7} " + " ".join(f"{y}(L)  {y}(M)" for y in fut),flush=True)
    for c in range(N):
        row=f"{CLASSES[c]:12s} {100*series[-1,c]:6.1f}% "
        for k in range(a.predict): row+=f"{100*lstm[k,c]:5.1f}%{100*markov[k,c]:6.1f}% "
        print(row,flush=True)
    print("\n(L)=LSTM forecast [course lectures 10-11]   (M)=Markov baseline [classical]",flush=True)
    print("Illustrative 'if current trends continue' projection — not a reliable prediction.",flush=True)
    print(flush=True)

    with open(f"{OUT}/forecast.csv","w") as f:
        f.write("class,"+",".join(str(y) for y in years)+","+",".join(f"{y}_LSTM,{y}_Markov" for y in fut)+"\n")
        for c in range(N):
            f.write(CLASSES[c]+","+",".join(f"{100*series[t,c]:.2f}" for t in range(T))+","
                    +",".join(f"{100*lstm[k,c]:.2f},{100*markov[k,c]:.2f}" for k in range(a.predict))+"\n")

    plt.figure(figsize=(13,7))
    for c in range(N):
        plt.plot(years,100*series[:,c],"-o",color=PALETTE[c],lw=2.4,label=CLASSES[c])
        plt.plot([years[-1]]+fut,[100*series[-1,c]]+list(100*lstm[:,c]),"--",color=PALETTE[c],lw=2,alpha=.9)
        plt.plot([years[-1]]+fut,[100*series[-1,c]]+list(100*markov[:,c]),":",color=PALETTE[c],lw=2,alpha=.7)
    plt.axvline(years[-1],color="grey",ls="--",alpha=.5)
    plt.text(years[-1]+0.05,plt.ylim()[1]*0.95,"forecast →",color="grey")
    plt.xlabel("year"); plt.ylabel("% of area"); plt.grid(alpha=.3)
    plt.title("Land-cover forecast — solid: observed · dashed: LSTM (lec 10-11) · dotted: Markov baseline")
    plt.legend(ncol=4,fontsize=9); plt.tight_layout(); plt.savefig(f"{OUT}/forecast.png",dpi=130); plt.close()
    print(f"\nSaved: {OUT}/forecast.png, {OUT}/forecast.csv",flush=True)


if __name__=="__main__":
    main()
