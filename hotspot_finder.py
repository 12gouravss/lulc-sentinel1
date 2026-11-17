import os, argparse, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import ee, geemap, rasterio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.colors import ListedColormap

PROJECT = "lulc-sentinel-513912"
device = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Water","Trees","Grassland","Cropland","Built-up","Bare/Sparse","Wetland"]
PALETTE = ["#1f78b4","#33a02c","#b2df8a","#fdbf6f","#e31a1c","#b15928","#6a3d9a"]
N = len(CLASSES); CMAP = ListedColormap(PALETTE); PATCH = 256; IN_CH = 10; OUT = "hotspot_out"


# multi-temporal U-Net (train_multitemporal.py for weight loading) 
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


def s1_month(aoi, year, month):
    start=ee.Date.fromYMD(year,month,1); end=start.advance(1,"month")
    col=(ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(start,end)
         .filter(ee.Filter.eq("instrumentMode","IW"))
         .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV"))
         .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VH"))
         .filter(ee.Filter.eq("orbitProperties_pass","DESCENDING")).select(["VV","VH"]))
    img=col.median().focalMedian(1.5,"circle","pixels")
    vv=img.select("VV").clamp(-25,0).unitScale(-25,0); vh=img.select("VH").clamp(-30,-5).unitScale(-30,-5)
    return vv.addBands(vh).rename(["VV","VH"]).toFloat()

def s1_stack(aoi, year):
    imgs=[s1_month(aoi,year,m).rename([f"VV{m:02d}",f"VH{m:02d}"]) for m in range(1,13)]
    return ee.Image.cat(imgs).toFloat()

def download(image, aoi, path):
    if not os.path.exists(path):
        geemap.download_ee_image(image.clip(aoi), path, region=aoi, scale=10, crs="EPSG:4326")

def features_from_stack(raw):            
    vv=raw[0::2]; vh=raw[1::2]; d=vv-vh  
    return np.stack([vv.mean(0),vv.std(0),vv.min(0),vv.max(0),
                     vh.mean(0),vh.std(0),vh.min(0),vh.max(0),
                     d.mean(0),d.std(0)],0).astype(np.float32)

@torch.no_grad()
def classify(model, feats):              
    a=np.nan_to_num(feats,nan=0.0); _,h,w=a.shape
    a=np.pad(a,((0,0),(0,(-h)%PATCH),(0,(-w)%PATCH)),mode="reflect"); _,H,W=a.shape
    out=np.zeros((H,W),np.int64)
    for iy in range(0,H,PATCH):
        for ix in range(0,W,PATCH):
            t=torch.from_numpy(a[:,iy:iy+PATCH,ix:ix+PATCH]).float().unsqueeze(0).to(device)
            out[iy:iy+PATCH,ix:ix+PATCH]=model(t).argmax(1)[0].cpu().numpy()
    return out[:h,:w]

def annual_map(model, aoi, year):
    tif=f"{OUT}/stack_{year}.tif"; download(s1_stack(aoi,year),aoi,tif)
    with rasterio.open(tif) as r: raw=r.read().astype(np.float32)
    return classify(model, features_from_stack(raw))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lat",type=float,required=True); ap.add_argument("--lon",type=float,required=True)
    ap.add_argument("--size-km",type=float,default=40); ap.add_argument("--grid",type=int,default=6)
    ap.add_argument("--year-a",type=int,default=2019); ap.add_argument("--year-b",type=int,default=2023)
    ap.add_argument("--weights",default="unet_multitemporal.pth")
    a=ap.parse_args(); os.makedirs(OUT,exist_ok=True)
    ee.Initialize(project=PROJECT)
    aoi=ee.Geometry.Point([a.lon,a.lat]).buffer(a.size_km*1000/2).bounds()
    b=aoi.coordinates().getInfo()[0]; lons=[p[0] for p in b]; lats=[p[1] for p in b]
    minlon,maxlon,minlat,maxlat=min(lons),max(lons),min(lats),max(lats)

    if not os.path.exists(a.weights):
        print(f"ERROR: {a.weights} not found — run train_multitemporal.py first."); return
    model=UNet().to(device); model.load_state_dict(torch.load(a.weights,map_location=device)); model.eval()
    print(f"Scanning {a.size_km} km around ({a.lat},{a.lon}) | grid {a.grid}x{a.grid} | "
          f"stable annual maps {a.year_a} vs {a.year_b}", flush=True)

    print(f"  building {a.year_a} annual map…", flush=True); pa=annual_map(model,aoi,a.year_a)
    print(f"  building {a.year_b} annual map…", flush=True); pb=annual_map(model,aoi,a.year_b)
    H=min(pa.shape[0],pb.shape[0]); W=min(pa.shape[1],pb.shape[1]); pa=pa[:H,:W]; pb=pb[:H,:W]
    changed=(pa!=pb)

    G=a.grid; gh,gw=H//G,W//G; score=np.zeros((G,G)); rows=[]
    for i in range(G):
        for j in range(G):
            s=100*changed[i*gh:(i+1)*gh, j*gw:(j+1)*gw].mean(); score[i,j]=s
            clat=maxlat-(i+0.5)/G*(maxlat-minlat); clon=minlon+(j+0.5)/G*(maxlon-minlon)
            rows.append((s,clat,clon,i,j))
    rows.sort(reverse=True)

    print("\n CHANGE HOTSPOTS (ranked, stable annual maps) ", flush=True)
    print(f"{'rank':>4} {'change%':>8} {'lat':>9} {'lon':>9}", flush=True)
    for k,(s,clat,clon,i,j) in enumerate(rows[:8],1):
        print(f"{k:>4} {s:>7.1f}% {clat:>9.4f} {clon:>9.4f}", flush=True)
    top=rows[0]
    print(f"\nTop hotspot: {top[0]:.1f}% changed at ({top[1]:.4f}, {top[2]:.4f})", flush=True)
    print(f"-> inspect it: python3 prototype.py --lat {top[1]:.4f} --lon {top[2]:.4f} --start-month 1 --year {a.year_b}", flush=True)
    print(f"(whole-area change: {100*changed.mean():.1f}%)", flush=True)
    print( flush=True)
    with open(f"{OUT}/hotspots.csv","w") as f:
        f.write("rank,change_pct,lat,lon\n")
        for k,(s,clat,clon,i,j) in enumerate(rows,1): f.write(f"{k},{s:.2f},{clat:.5f},{clon:.5f}\n")

    fig,ax=plt.subplots(1,2,figsize=(14,6))
    im=ax[0].imshow(score,cmap="inferno",extent=[minlon,maxlon,minlat,maxlat],aspect="auto")
    ax[0].set_title(f"Change intensity grid ({a.year_a}→{a.year_b})",fontsize=13)
    ax[0].set_xlabel("lon"); ax[0].set_ylabel("lat"); fig.colorbar(im,ax=ax[0],fraction=0.046,label="% changed")
    ti,tj=top[3],top[4]
    ax[0].add_patch(plt.Rectangle((minlon+tj/G*(maxlon-minlon), maxlat-(ti+1)/G*(maxlat-minlat)),
        (maxlon-minlon)/G,(maxlat-minlat)/G, fill=False, edgecolor="cyan", lw=3))
    ax[1].imshow(changed,cmap="Reds",vmin=0,vmax=1); ax[1].set_title(f"Changed pixels ({100*changed.mean():.1f}%)",fontsize=13); ax[1].axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/change_hotspots.png",dpi=130); plt.close()

    fig,ax=plt.subplots(1,3,figsize=(17,5.5))
    ax[0].imshow(pa,cmap=CMAP,vmin=0,vmax=N-1); ax[0].set_title(f"LULC {a.year_a}",fontsize=13); ax[0].axis("off")
    ax[1].imshow(pb,cmap=CMAP,vmin=0,vmax=N-1); ax[1].set_title(f"LULC {a.year_b}",fontsize=13); ax[1].axis("off")
    rgb=np.ones((*changed.shape,3))*0.85
    for c in range(N): rgb[changed&(pb==c)]=mc.to_rgb(PALETTE[c])
    ax[2].imshow(rgb); ax[2].set_title(f"Change by new class ({a.year_b})",fontsize=13); ax[2].axis("off")
    fig.legend([plt.Rectangle((0,0),1,1,color=PALETTE[c]) for c in range(N)],CLASSES,loc="lower center",ncol=7,fontsize=11)
    plt.tight_layout(rect=[0,0.05,1,1]); plt.savefig(f"{OUT}/top_hotspot_maps.png",dpi=130); plt.close()
    print(f"\nSaved: {OUT}/change_hotspots.png, {OUT}/top_hotspot_maps.png, {OUT}/hotspots.csv", flush=True)


if __name__ == "__main__":
    main()
