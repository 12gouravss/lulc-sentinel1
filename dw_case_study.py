import os, argparse, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import ee, geemap, rasterio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

PROJECT="lulc-sentinel-513912"; device="cuda" if torch.cuda.is_available() else "cpu"
CLASSES=["Water","Trees","Grassland","Cropland","Built-up","Bare/Sparse","Wetland"]
PALETTE=["#1f78b4","#33a02c","#b2df8a","#fdbf6f","#e31a1c","#b15928","#6a3d9a"]
N=len(CLASSES); PATCH=256; SCALE=50; OUT="case_study_out"


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

def s1_month(aoi,year,month):
    st=ee.Date.fromYMD(year,month,1); en=st.advance(1,"month")
    col=(ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(st,en)
         .filter(ee.Filter.eq("instrumentMode","IW"))
         .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV"))
         .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VH"))
         .filter(ee.Filter.eq("orbitProperties_pass","DESCENDING")).select(["VV","VH"]))
    img=col.median().focalMedian(1.5,"circle","pixels")
    med=ee.Image(ee.Algorithms.If(img.bandNames().size().gt(0),img,ee.Image([-30.0,-30.0]).rename(["VV","VH"])))
    vv=med.select("VV").clamp(-25,0).unitScale(-25,0); vh=med.select("VH").clamp(-30,-5).unitScale(-30,-5)
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


DW2OURS={0:0,1:1,2:2,3:6,4:3,5:2,6:4,7:5,8:5}
def dw_annual(aoi,year):
    col=(ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
         .filterBounds(aoi).filterDate(f"{year}-01-01",f"{year}-12-31").select("label"))
    return col.mode()   # most frequent class per pixel over the year

def pct(maparr):
    v=maparr[np.isfinite(maparr)].astype(int); v=v[(v>=0)&(v<N)]
    c=np.bincount(v,minlength=N); return 100*c/max(c.sum(),1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lat",type=float,required=True); ap.add_argument("--lon",type=float,required=True)
    ap.add_argument("--city",default="location"); ap.add_argument("--year",type=int,default=2021)
    ap.add_argument("--size-km",type=float,default=12); ap.add_argument("--weights",default="unet_multitemporal.pth")
    a=ap.parse_args(); os.makedirs(OUT,exist_ok=True)
    ee.Initialize(project=PROJECT)
    aoi=ee.Geometry.Point([a.lon,a.lat]).buffer(a.size_km*1000/2).bounds()

    
    model=UNetMT(10).to(device); model.load_state_dict(torch.load(a.weights,map_location=device)); model.eval()
    s1tif=f"{OUT}/{a.city}_s1_{a.year}.tif"; print("  building radar stack…",flush=True)
    download(s1_stack(aoi,a.year),aoi,s1tif)
    with rasterio.open(s1tif) as r: raw=r.read().astype(np.float32)
    ours=classify(model,features_from_stack(raw))

    
    dwtif=f"{OUT}/{a.city}_dw_{a.year}.tif"; print("  building Dynamic World reference…",flush=True)
    download(dw_annual(aoi,a.year).rename("label").toFloat(),aoi,dwtif)
    with rasterio.open(dwtif) as r: dwraw=r.read(1)
    dw=np.vectorize(lambda v: DW2OURS.get(int(v),2))(np.nan_to_num(dwraw,nan=2)).astype(int)

    po=pct(ours); pd=pct(dw)
    print(f" CASE STUDY: {a.city} ({a.year}) ", flush=True)
    print(f"{'Class':12s} {'Ours %':>8s} {'DynWorld %':>11s}",flush=True)
    for c in range(N): print(f"{CLASSES[c]:12s} {po[c]:8.1f} {pd[c]:11.1f}",flush=True)
    print(f"\nKey urban checks:",flush=True)
    print(f"  Built-up : ours {po[4]:.1f}%  vs  Dynamic World {pd[4]:.1f}%",flush=True)
    print(f"  Water    : ours {po[0]:.1f}%  vs  Dynamic World {pd[0]:.1f}%",flush=True)
    print("Independent reference = Google Dynamic World (optical, not used in training).",flush=True)
    print(flush=True)

    cmap=ListedColormap(PALETTE)
    fig,ax=plt.subplots(1,2,figsize=(13,6))
    ax[0].imshow(ours,cmap=cmap,vmin=0,vmax=N-1); ax[0].set_title(f"Our model — {a.city} {a.year}"); ax[0].axis("off")
    ax[1].imshow(dw,cmap=cmap,vmin=0,vmax=N-1); ax[1].set_title("Dynamic World (independent)"); ax[1].axis("off")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=PALETTE[i],label=CLASSES[i]) for i in range(N)],
               loc="lower center",ncol=7,fontsize=8,frameon=False)
    plt.tight_layout(rect=[0,0.05,1,1]); plt.savefig(f"{OUT}/{a.city}_vs_dynamicworld.png",dpi=140); plt.close()
    print(f"\nSaved: {OUT}/{a.city}_vs_dynamicworld.png",flush=True)

if __name__=="__main__":
    main()
