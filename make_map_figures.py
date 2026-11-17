import os, argparse, numpy as np, torch
import ee, geemap, rasterio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from train import UNet, device, N, CLASSES          # single-month U-Net (in_ch=2)

PROJECT="lulc-sentinel-513912"
PALETTE=["#1f78b4","#33a02c","#b2df8a","#fdbf6f","#e31a1c","#b15928","#6a3d9a"]
PATCH=256; SCALE=50; OUT="map_figs"; CMAP=ListedColormap(PALETTE)

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
    return vv.addBands(vh).rename([f"VV{month:02d}",f"VH{month:02d}"]).toFloat()

@torch.no_grad()
def classify2(model,arr2):
    a=np.nan_to_num(arr2,nan=0.0); _,h,w=a.shape
    a=np.pad(a,((0,0),(0,(-h)%PATCH),(0,(-w)%PATCH)),mode="reflect"); _,H,W=a.shape; out=np.zeros((H,W),np.int64)
    for iy in range(0,H,PATCH):
        for ix in range(0,W,PATCH):
            t=torch.from_numpy(a[:,iy:iy+PATCH,ix:ix+PATCH]).float().unsqueeze(0).to(device)
            out[iy:iy+PATCH,ix:ix+PATCH]=model(t).argmax(1)[0].cpu().numpy()
    return out[:h,:w]

def pct(m):
    c=np.bincount(m.ravel(),minlength=N); return 100*c/max(c.sum(),1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lat",type=float,required=True); ap.add_argument("--lon",type=float,required=True)
    ap.add_argument("--city",default="location"); ap.add_argument("--year",type=int,default=2021)
    ap.add_argument("--size-km",type=float,default=12); ap.add_argument("--weights",default="unet_s1_lulc.pth")
    ap.add_argument("--test-tile",default="tiles/catalonia_2021_06.tif")
    a=ap.parse_args(); os.makedirs(OUT,exist_ok=True)
    ee.Initialize(project=PROJECT)
    model=UNet(in_ch=2,n_classes=N,base=64).to(device); model.load_state_dict(torch.load(a.weights,map_location=device)); model.eval()
    aoi=ee.Geometry.Point([a.lon,a.lat]).buffer(a.size_km*1000/2).bounds()

    #  12 monthly maps 
    stack=ee.Image.cat([s1_month(aoi,a.year,m) for m in range(1,13)]).toFloat()
    tif=f"{OUT}/{a.city}_stack_{a.year}.tif"
    if not os.path.exists(tif):
        print("  building 12-month radar stack…",flush=True)
        geemap.download_ee_image(stack.clip(aoi),tif,region=aoi,scale=SCALE,crs="EPSG:4326")
    with rasterio.open(tif) as r: raw=r.read().astype(np.float32)   # (24,H,W)

    maps=[]; months=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig,ax=plt.subplots(3,4,figsize=(14,10))
    for m in range(12):
        mp=classify2(model,raw[2*m:2*m+2]); maps.append(mp)
        ax[m//4,m%4].imshow(mp,cmap=CMAP,vmin=0,vmax=N-1); ax[m//4,m%4].set_title(months[m]); ax[m//4,m%4].axis("off")
    fig.suptitle(f"Monthly LULC maps — {a.city} {a.year}",fontsize=14)
    fig.legend(handles=[Patch(color=PALETTE[i],label=CLASSES[i]) for i in range(N)],
               loc="lower center",ncol=7,fontsize=9,frameon=False)
    plt.tight_layout(rect=[0,0.04,1,0.97]); plt.savefig(f"{OUT}/monthly_maps_{a.city}.png",dpi=130); plt.close()

    # per-class % and change
    print(f"\n {a.city} {a.year} — per-class % by month ",flush=True)
    print("Month  " + " ".join(f"{c[:4]:>5s}" for c in CLASSES),flush=True)
    series=[]
    for m in range(12):
        p=pct(maps[m]); series.append(p)
        print(f"{months[m]:5s}  " + " ".join(f"{v:5.1f}" for v in p),flush=True)
    series=np.array(series)
    tot=series[-1]-series[0]
    print("\nTotal change (Dec - Jan), percentage points:",flush=True)
    print("       " + " ".join(f"{v:+5.1f}" for v in tot),flush=True)

    # prediction vs ground truth 
    if os.path.exists(a.test_tile):
        with rasterio.open(a.test_tile) as r: t=r.read().astype(np.float32)   # (3,H,W): VV,VH,label
        H,W=t.shape[1:]; i=(H//2)//PATCH*PATCH; j=(W//2)//PATCH*PATCH
        x=t[:2, i:i+PATCH, j:j+PATCH]; truth=np.clip(np.nan_to_num(t[2, i:i+PATCH, j:j+PATCH]),0,N-1).astype(int)
        pred=classify2(model,x)
        fig,ax=plt.subplots(1,3,figsize=(14,5))
        ax[0].imshow(x[0],cmap="gray"); ax[0].set_title("Radar input (VV)"); ax[0].axis("off")
        ax[1].imshow(pred,cmap=CMAP,vmin=0,vmax=N-1); ax[1].set_title("Model prediction"); ax[1].axis("off")
        ax[2].imshow(truth,cmap=CMAP,vmin=0,vmax=N-1); ax[2].set_title("WorldCover truth"); ax[2].axis("off")
        fig.legend(handles=[Patch(color=PALETTE[i],label=CLASSES[i]) for i in range(N)],
                   loc="lower center",ncol=7,fontsize=9,frameon=False)
        plt.tight_layout(rect=[0,0.06,1,1]); plt.savefig(f"{OUT}/prediction_vs_truth.png",dpi=140); plt.close()
        print(f"\nSaved: {OUT}/monthly_maps_{a.city}.png, {OUT}/prediction_vs_truth.png",flush=True)
    else:
        print(f"\n(test tile {a.test_tile} not found — only monthly_maps saved)",flush=True)
        print(f"Saved: {OUT}/monthly_maps_{a.city}.png",flush=True)

if __name__=="__main__":
    main()
