import os, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import ee, geemap, rasterio, gradio as gr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.colors import ListedColormap

PROJECT="lulc-sentinel-513912"
device="cuda" if torch.cuda.is_available() else "cpu"
CLASSES=["Water","Trees","Grassland","Cropland","Built-up","Bare/Sparse","Wetland"]
PALETTE=["#1f78b4","#33a02c","#b2df8a","#fdbf6f","#e31a1c","#b15928","#6a3d9a"]
N=len(CLASSES); CMAP=ListedColormap(PALETTE); CMAP.set_bad("white"); PATCH=256; WIN_KM=4.0; HOTSPOT_SCALE=60
MONTH_NAMES=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
PRESETS={
 "Mannheim — urban (49.49, 8.47)":(49.4875,8.4660,"mannheim"),
 "Flevoland — farmland (52.52, 5.50)":(52.52,5.50,"flevoland"),
 "Po Valley — cropland (45.25, 9.12)":(45.25,9.12,"povalley"),
 "Berlin — city (52.52, 13.40)":(52.52,13.405,"berlin"),
 "Rhine-Neckar — mixed (49.40, 8.68)":(49.40,8.68,"rhineneckar"),
 "Almería — arid (36.84, -2.43)":(36.84,-2.43,"almeria"),
 "Danube bend — river (47.80, 18.95)":(47.80,18.95,"danube"),
 "Custom (use the boxes below)":None,
}


class DoubleConv(nn.Module):                      
    def __init__(s,a,b):
        super().__init__(); s.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(True),
                                                nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(True))
    def forward(s,x): return s.net(x)
class UNet(nn.Module):
    def __init__(s,in_ch=2,n=7,base=64):
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
class DCb(nn.Module):                             # multi-temporal (train_multitemporal naming: .n)
    def __init__(s,a,b):
        super().__init__(); s.n=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(True),
                                              nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(True))
    def forward(s,x): return s.n(x)
class UNetMT(nn.Module):
    def __init__(s,in_ch=10,n=7,base=64):
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
class TemporalLSTM(nn.Module):
    def __init__(s,in_dim=7,hidden=64,n=7,layers=1):
        super().__init__(); s.lstm=nn.LSTM(in_dim,hidden,num_layers=layers,batch_first=True,bidirectional=True)
        s.head=nn.Linear(2*hidden,n)
    def forward(s,x): h,_=s.lstm(x); return s.head(h)
class ForecastLSTM(nn.Module):                          
    def __init__(s,dim=7,hidden=24):
        super().__init__(); s.lstm=nn.LSTM(dim,hidden,batch_first=True); s.head=nn.Linear(hidden,dim)
    def forward(s,x): h,_=s.lstm(x); return s.head(h[:,-1])


def s1_month(aoi,year,month):
    st=ee.Date.fromYMD(year,month,1); en=st.advance(1,"month")
    col=(ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(st,en)
         .filter(ee.Filter.eq("instrumentMode","IW"))
         .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV"))
         .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VH"))
         .filter(ee.Filter.eq("orbitProperties_pass","DESCENDING")).select(["VV","VH"]))
    med=col.median()                                     # empty when a month has no data
    med=ee.Image(ee.Algorithms.If(med.bandNames().size().gt(0),med,ee.Image([-30.0,-30.0]).rename(["VV","VH"])))
    img=med.focalMedian(1.5,"circle","pixels")
    vv=img.select("VV").clamp(-25,0).unitScale(-25,0); vh=img.select("VH").clamp(-30,-5).unitScale(-30,-5)
    return vv.addBands(vh).rename(["VV","VH"]).toFloat()
def s1_stack(aoi,year,start_month=1):
    return ee.Image.cat([s1_month(aoi,year+(start_month-1+i)//12,(start_month-1+i)%12+1).rename([f"VV{i:02d}",f"VH{i:02d}"]) for i in range(12)]).toFloat()
def download(image,aoi,path,scale=10):
    if not os.path.exists(path): geemap.download_ee_image(image.clip(aoi),path,region=aoi,scale=scale,crs="EPSG:4326")
def dynamic_world(aoi,year):
    st=ee.Date.fromYMD(year,1,1); en=ee.Date.fromYMD(year,12,31)
    dw=(ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(aoi).filterDate(st,en).select("label").mode())
    return dw.remap([0,1,2,3,4,5,6,7,8],[0,1,2,6,3,2,4,5,5]).rename("lulc").toInt()
def pad(a): _,h,w=a.shape; return np.pad(a,((0,0),(0,(-h)%PATCH),(0,(-w)%PATCH)),mode="reflect"),h,w
@torch.no_grad()
def unet_probs(unet,arr):
    a,h,w=pad(np.nan_to_num(arr,nan=0.0)); _,H,W=a.shape; out=np.zeros((N,H,W),np.float32)
    for iy in range(0,H,PATCH):
        for ix in range(0,W,PATCH):
            t=torch.from_numpy(a[:,iy:iy+PATCH,ix:ix+PATCH]).float().unsqueeze(0).to(device)
            out[:,iy:iy+PATCH,ix:ix+PATCH]=F.softmax(unet(t),1)[0].cpu().numpy()
    return out[:,:h,:w]
@torch.no_grad()
def classify_mt(feats):
    a,h,w=pad(np.nan_to_num(feats,nan=0.0)); _,H,W=a.shape; out=np.zeros((H,W),np.int64)
    for iy in range(0,H,PATCH):
        for ix in range(0,W,PATCH):
            t=torch.from_numpy(a[:,iy:iy+PATCH,ix:ix+PATCH]).float().unsqueeze(0).to(device)
            out[iy:iy+PATCH,ix:ix+PATCH]=UNET_MT(t).argmax(1)[0].cpu().numpy()
    return out[:h,:w]
def features_from_stack(raw):
    raw=np.nan_to_num(raw,nan=0.0)
    vv=raw[0::2]; vh=raw[1::2]; d=vv-vh
    return np.stack([vv.mean(0),vv.std(0),vv.min(0),vv.max(0),vh.mean(0),vh.std(0),vh.min(0),vh.max(0),d.mean(0),d.std(0)],0).astype(np.float32)
@torch.no_grad()
def lstm_refine(p):
    T,C,H,W=p.shape; seq=torch.from_numpy(p.transpose(2,3,0,1).reshape(-1,T,C)).float(); outs=[]
    for b in range(0,len(seq),200000): outs.append(LSTM(seq[b:b+200000].to(device)).argmax(-1).cpu())
    return torch.cat(outs).numpy().reshape(H,W,T).transpose(2,0,1)
def class_pct(m,mask=None):
    v=m[mask] if mask is not None else m
    c=np.bincount(v.ravel(),minlength=N).astype(float); return 100*c/max(c.sum(),1)


UNET=UNet(base=64).to(device); UNET.load_state_dict(torch.load("unet_s1_lulc.pth",map_location=device)); UNET.eval()
LSTM=TemporalLSTM().to(device); LSTM.load_state_dict(torch.load("lstm_temporal.pth",map_location=device)); LSTM.eval()
UNET_MT=UNetMT(10).to(device); UNET_MT.load_state_dict(torch.load("unet_multitemporal.pth",map_location=device)); UNET_MT.eval()
ee.Initialize(project=PROJECT)


def run_region(lat,lon,start_month,year,out,progress=None):
    os.makedirs(out,exist_ok=True)
    aoi=ee.Geometry.Point([lon,lat]).buffer(WIN_KM*1000/2).bounds()
    months=[(start_month-1+i)%12+1 for i in range(12)]
    stk=f"{out}/s1stack_{year}_m{start_month}.tif"
    if not os.path.exists(stk):
        if progress: progress(0.15,desc="Fetching 12 months of Sentinel-1…")
        download(s1_stack(aoi,year,start_month),aoi,stk)
    with rasterio.open(stk) as r: raw=r.read().astype(np.float32)
    valid=np.isfinite(raw).all(0)&(np.abs(raw)>1e-6).any(0)     
    M=lambda m: np.ma.masked_where(~valid,m)
    probs=[]
    for i in range(12):
        if progress: progress(0.4+0.45*i/12,desc=f"Classifying month {i+1}/12")
        probs.append(unet_probs(UNET,raw[2*i:2*i+2]))
    if progress: progress(0.9,desc="Temporal LSTM + change analysis")
    probs=np.nan_to_num(np.stack(probs,0),nan=0.0); lstm_maps=lstm_refine(probs); conf=probs.max(1).mean(0)
    pct=np.stack([class_pct(lstm_maps[t],valid) for t in range(12)],0); total=pct[-1]-pct[0]
    tm=np.zeros((N,N),float)
    for s,d in zip(lstm_maps[0].ravel(),lstm_maps[-1].ravel()): tm[s,d]+=1
    tm_pct=100*tm/tm.sum()
    dwp=f"{out}/dynamic_world.tif"; download(dynamic_world(aoi,year),aoi,dwp)
    with rasterio.open(dwp) as r: dw=np.clip(np.nan_to_num(r.read(1),nan=0),0,N-1).astype(int)
    dwp_=class_pct(dw,valid); ourp=class_pct(lstm_maps[0],valid); agree=100-0.5*np.abs(dwp_-ourp).sum()

    def leg(f): f.legend([plt.Rectangle((0,0),1,1,color=PALETTE[c]) for c in range(N)],CLASSES,loc="lower center",ncol=7,fontsize=11)
    annual=np.stack([(lstm_maps==c).sum(0) for c in range(N)],0).argmax(0)   # most-common class per pixel (clean, no flicker)
    ap=class_pct(annual,valid)
    fig=plt.figure(figsize=(16,13)); gs=fig.add_gridspec(4,4,height_ratios=[1.7,1,1,1],hspace=0.30,wspace=0.08)
    axv=fig.add_subplot(gs[0,0]); axv.imshow(raw[0],cmap="gray"); axv.set_title("Radar input (VV)",fontsize=13); axv.axis("off")
    axa=fig.add_subplot(gs[0,1:3]); axa.imshow(M(annual),cmap=CMAP,vmin=0,vmax=N-1); axa.set_title("Annual land cover — most common class (LSTM-refined)",fontsize=14); axa.axis("off")
    axb=fig.add_subplot(gs[0,3]); axb.barh(range(N),ap,color=PALETTE); axb.invert_yaxis(); axb.set_yticks(range(N)); axb.set_yticklabels(CLASSES,fontsize=9); axb.set_title("% of area",fontsize=12); axb.spines[["top","right"]].set_visible(False)
    for i,v in enumerate(ap): axb.text(v+0.5,i,f"{v:.0f}%",va="center",fontsize=8)
    for t in range(12):
        axm=fig.add_subplot(gs[1+t//4,t%4]); axm.imshow(M(lstm_maps[t]),cmap=CMAP,vmin=0,vmax=N-1); axm.set_title(f"{MONTH_NAMES[months[t]-1]} {year+(start_month-1+t)//12}",fontsize=11); axm.axis("off")
    fig.legend([plt.Rectangle((0,0),1,1,color=PALETTE[c]) for c in range(N)],CLASSES,loc="lower center",ncol=7,fontsize=11)
    fig.suptitle("Land-cover overview — radar input, annual map, and all 12 months",fontsize=16)
    plt.tight_layout(rect=[0,0.03,1,0.96]); f1=f"{out}/monthly_maps.png"; plt.savefig(f1,dpi=120); plt.close()
    fig,ax=plt.subplots(figsize=(12,6))
    for c in range(N): ax.plot(range(1,13),pct[:,c],marker="o",lw=2.5,color=PALETTE[c],label=CLASSES[c])
    ax.set_xticks(range(1,13)); ax.set_xticklabels([MONTH_NAMES[m-1] for m in months]); ax.set_ylabel("% of area"); ax.set_title("Class area % across 12 months"); ax.grid(alpha=.3); ax.legend(ncol=4,fontsize=10)
    plt.tight_layout(); f2=f"{out}/class_percent_over_months.png"; plt.savefig(f2,dpi=120); plt.close()
    hh1=f"{MONTH_NAMES[months[0]-1]} {year}"; hh2=f"{MONTH_NAMES[months[-1]-1]} {year+(start_month-1+11)//12}"
    changed=(lstm_maps[0]!=lstm_maps[-1])&valid; pc=100*changed.sum()/max(valid.sum(),1)
    base=np.zeros((*changed.shape,3))
    for c in range(N): base[(lstm_maps[-1]==c)&valid]=mc.to_rgb(PALETTE[c])
    hl=base*0.22; hl[changed]=base[changed]; hl[~valid]=1.0       # dim land, light up changes, blank nodata
    fig,ax=plt.subplots(1,3,figsize=(17,6))
    ax[0].imshow(M(lstm_maps[0]),cmap=CMAP,vmin=0,vmax=N-1); ax[0].set_title(f"Land cover — {hh1}"); ax[0].axis("off")
    ax[1].imshow(M(lstm_maps[-1]),cmap=CMAP,vmin=0,vmax=N-1); ax[1].set_title(f"Land cover — {hh2}"); ax[1].axis("off")
    ax[2].imshow(hl); ax[2].set_title(f"What changed ({pc:.1f}% of area) — lit by its new class"); ax[2].axis("off"); leg(fig)
    plt.tight_layout(rect=[0,0.05,1,1]); f3=f"{out}/change_map.png"; plt.savefig(f3,dpi=120); plt.close()
    low=(conf<0.5)&valid
    cmapc=plt.cm.RdYlGn.copy(); cmapc.set_bad("#e9e9e9")
    fig,ax=plt.subplots(1,2,figsize=(14,6))
    im=ax[0].imshow(M(conf),cmap=cmapc,vmin=0.3,vmax=1.0); ax[0].set_title("Model confidence  —  green = sure, red = unsure"); ax[0].axis("off")
    cb=fig.colorbar(im,ax=ax[0],fraction=0.046); cb.set_label("certainty (max class probability)")
    ax[1].imshow(np.ma.masked_where(~valid,low.astype(float)),cmap="Reds",vmin=0,vmax=1); ax[1].set_title(f"Uncertain pixels <0.5  —  {100*low.sum()/max(valid.sum(),1):.1f}% of area"); ax[1].axis("off")
    plt.tight_layout(); f4=f"{out}/confidence_map.png"; plt.savefig(f4,dpi=120); plt.close()
    fig,ax=plt.subplots(figsize=(8.5,7.5)); im=ax.imshow(tm_pct,cmap="inferno")
    ax.set_xticks(range(N)); ax.set_yticks(range(N)); ax.set_xticklabels(CLASSES,rotation=45,ha="right"); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("to (month 12)"); ax.set_ylabel("from (month 1)"); ax.set_title("Class transitions, month 1 → 12 (%)")
    for s in range(N):
        for d in range(N):
            if tm_pct[s,d]>=0.1: ax.text(d,s,f"{tm_pct[s,d]:.1f}",ha="center",va="center",color="white" if tm_pct[s,d]<tm_pct.max()*0.6 else "black",fontsize=8)
    fig.colorbar(im,ax=ax,fraction=0.046); plt.tight_layout(); f5=f"{out}/transition_matrix.png"; plt.savefig(f5,dpi=120); plt.close()
    fig,ax=plt.subplots(1,3,figsize=(17,5.5))
    ax[0].imshow(M(lstm_maps[0]),cmap=CMAP,vmin=0,vmax=N-1); ax[0].set_title("Ours (month 1)"); ax[0].axis("off")
    ax[1].imshow(M(dw),cmap=CMAP,vmin=0,vmax=N-1); ax[1].set_title("Google Dynamic World"); ax[1].axis("off")
    x=np.arange(N); ax[2].bar(x-0.2,ourp,0.4,label="Ours",color="#1f78b4"); ax[2].bar(x+0.2,dwp_,0.4,label="Dynamic World",color="#e31a1c")
    ax[2].set_xticks(x); ax[2].set_xticklabels(CLASSES,rotation=45,ha="right"); ax[2].set_ylabel("% of area"); ax[2].set_title(f"Agreement: {agree:.1f}%"); ax[2].legend()
    plt.tight_layout(); f6=f"{out}/sanity_check_dynamic_world.png"; plt.savefig(f6,dpi=120); plt.close()

    end_m=months[-1]; end_y=year+(start_month-1+11)//12; h1=f"{MONTH_NAMES[months[0]-1]} {year}"; h2=f"{MONTH_NAMES[end_m-1]} {end_y}"
    md=f"**{lat}, {lon}** · {WIN_KM} km · {h1} → {h2}  _(maps are temporal-LSTM refined)_\n\n| Class | {h1} | {h2} | Change |\n|---|---|---|---|\n"
    for c in range(N):
        ch=0.0 if abs(total[c])<0.005 else total[c]; md+=f"| {CLASSES[c]} | {pct[0,c]:.1f}% | {pct[-1,c]:.1f}% | {ch:+.2f} pp |\n"
    flows=sorted([(tm_pct[s,d],s,d) for s in range(N) for d in range(N) if s!=d],reverse=True)[:5]
    md+="\n**Top transitions (LSTM-refined, month 1 → 12)**\n\n"+ "".join(f"- {CLASSES[s]} → {CLASSES[d]}: **{v:.2f}%**\n" for v,s,d in flows)
    top=flows[0]; stats={"changed":pc,"confidence":float(conf.mean()),"agreement":agree,"top":f"{CLASSES[top[1]][:5]}→{CLASSES[top[2]][:5]}","topv":top[0]}
    A=os.path.abspath
    return A(f1),A(f2),A(f3),A(f4),A(f5),A(f6),md,stats


def annual_map(aoi,year,out,progress,frac):
    tif=f"{out}/stack_{year}_s{HOTSPOT_SCALE}.tif"
    if not os.path.exists(tif):
        if progress: progress(frac,desc=f"Fetching 12 months of {year} (fast scan)…")
        download(s1_stack(aoi,year),aoi,tif,scale=HOTSPOT_SCALE)   # coarse resolution = fast
    with rasterio.open(tif) as r: raw=r.read().astype(np.float32)
    return classify_mt(features_from_stack(raw))
def run_hotspot(lat,lon,size_km,grid,year_a,year_b,progress=None):
    out=os.path.abspath(f"hotspot_out_{lat}_{lon}".replace(".","p").replace("-","m")); os.makedirs(out,exist_ok=True)
    aoi=ee.Geometry.Point([lon,lat]).buffer(size_km*1000/2).bounds()
    b=aoi.coordinates().getInfo()[0]; lons=[p[0] for p in b]; lats=[p[1] for p in b]
    minlon,maxlon,minlat,maxlat=min(lons),max(lons),min(lats),max(lats)
    pa=annual_map(aoi,int(year_a),out,progress,0.2); pb=annual_map(aoi,int(year_b),out,progress,0.6)
    if progress: progress(0.9,desc="Scoring grid…")
    H=min(pa.shape[0],pb.shape[0]); W=min(pa.shape[1],pb.shape[1]); pa=pa[:H,:W]; pb=pb[:H,:W]
    changed=pa!=pb; G=int(grid); gh,gw=H//G,W//G; score=np.zeros((G,G)); rows=[]
    for i in range(G):
        for j in range(G):
            s=100*changed[i*gh:(i+1)*gh,j*gw:(j+1)*gw].mean(); score[i,j]=s
            rows.append((s,maxlat-(i+0.5)/G*(maxlat-minlat),minlon+(j+0.5)/G*(maxlon-minlon),i,j))
    rows.sort(reverse=True); top=rows[0]
    fig,ax=plt.subplots(1,2,figsize=(14,6))
    im=ax[0].imshow(score,cmap="inferno",extent=[minlon,maxlon,minlat,maxlat],aspect="auto")
    ax[0].set_title(f"Change intensity grid ({year_a}→{year_b})"); ax[0].set_xlabel("lon"); ax[0].set_ylabel("lat"); fig.colorbar(im,ax=ax[0],fraction=0.046,label="% changed")
    ax[0].add_patch(plt.Rectangle((minlon+top[4]/G*(maxlon-minlon),maxlat-(top[3]+1)/G*(maxlat-minlat)),(maxlon-minlon)/G,(maxlat-minlat)/G,fill=False,edgecolor="cyan",lw=3))
    ax[1].imshow(changed,cmap="Reds",vmin=0,vmax=1); ax[1].set_title(f"Changed pixels ({100*changed.mean():.1f}%)"); ax[1].axis("off")
    plt.tight_layout(); g1=f"{out}/change_hotspots.png"; plt.savefig(g1,dpi=120); plt.close()
    fig,ax=plt.subplots(1,3,figsize=(17,5.5))
    ax[0].imshow(pa,cmap=CMAP,vmin=0,vmax=N-1); ax[0].set_title(f"LULC {year_a}"); ax[0].axis("off")
    ax[1].imshow(pb,cmap=CMAP,vmin=0,vmax=N-1); ax[1].set_title(f"LULC {year_b}"); ax[1].axis("off")
    rgb=np.ones((*changed.shape,3))*0.85
    for c in range(N): rgb[changed&(pb==c)]=mc.to_rgb(PALETTE[c])
    ax[2].imshow(rgb); ax[2].set_title(f"Change by new class ({year_b})"); ax[2].axis("off")
    fig.legend([plt.Rectangle((0,0),1,1,color=PALETTE[c]) for c in range(N)],CLASSES,loc="lower center",ncol=7,fontsize=11)
    plt.tight_layout(rect=[0,0.05,1,1]); g2=f"{out}/top_hotspot_maps.png"; plt.savefig(g2,dpi=120); plt.close()
    md=f"**Top change hotspots** · {size_km} km · {year_a} → {year_b} · whole-area change **{100*changed.mean():.1f}%**\n\n| rank | change % | lat | lon |\n|---|---|---|---|\n"
    for k,(s,clat,clon,i,j) in enumerate(rows[:6],1): md+=f"| {k} | {s:.1f}% | {clat:.4f} | {clon:.4f} |\n"
    md+=f"\n_Top hotspot: {top[0]:.1f}% changed at ({top[1]:.4f}, {top[2]:.4f}) — stable annual maps from the multi-temporal model, so this is real change, not single-date noise._"
    A=os.path.abspath; return A(g1),A(g2),md


TILES="""<div class="mrow">
<div class="metric m1"><div class="v">{changed:.1f}%</div><div class="l">Area changed</div></div>
<div class="metric m2"><div class="v">{confidence:.2f}</div><div class="l">Mean confidence</div></div>
<div class="metric m3"><div class="v">{agreement:.0f}%</div><div class="l">Dynamic World ag.</div></div>
<div class="metric m4"><div class="v" style="font-size:16px">{top}</div><div class="l">Top transition {topv:.1f}%</div></div></div>"""
def analyze(preset,lat,lon,start_month,year,progress=gr.Progress()):
    try:
        if PRESETS.get(preset) is not None: lat,lon,base=PRESETS[preset]
        else: base=f"custom_{lat}_{lon}".replace(".","p").replace("-","m")
        out=os.path.abspath(f"prototype_out_{base}_{int(year)}_m{int(start_month)}")
        f1,f2,f3,f4,f5,f6,md,s=run_region(float(lat),float(lon),int(start_month),int(year),out,progress=progress)
        return f1,f2,f3,f4,f5,f6,md,TILES.format(**s)
    except Exception:
        import traceback; e=f"### Error\n```\n{traceback.format_exc()[-1200:]}\n```"
        return None,None,None,None,None,None,e,""
def analyze_hotspot(lat,lon,size_km,grid,ya,yb,progress=gr.Progress()):
    try: return run_hotspot(float(lat),float(lon),float(size_km),int(grid),int(ya),int(yb),progress=progress)
    except Exception:
        import traceback; return None,None,f"### Error\n```\n{traceback.format_exc()[-1200:]}\n```"


def run_forecast(lat,lon,ystart,yend,predict,progress=None):
    out=os.path.abspath(f"forecast_out_{lat}_{lon}".replace(".","p").replace("-","m")); os.makedirs(out,exist_ok=True)
    aoi=ee.Geometry.Point([lon,lat]).buffer(20*1000/2).bounds()
    years=list(range(int(ystart),int(yend)+1)); maps=[]
    for i,y in enumerate(years): maps.append(annual_map(aoi,y,out,progress,0.1+0.7*i/max(len(years),1)))
    H=min(m.shape[0] for m in maps); W=min(m.shape[1] for m in maps); maps=[m[:H,:W] for m in maps]
    series=np.stack([np.bincount(m.ravel(),minlength=N)/m.size for m in maps],0); T=len(years); pr=int(predict)
    # Markov baseline (classical)
    P=np.zeros((N,N))
    for t in range(T-1):
        for c in range(N):
            msk=maps[t]==c
            if msk.sum()>0: P[c]+=np.bincount(maps[t+1][msk],minlength=N)/msk.sum()
    P/=max(T-1,1); P=P/P.sum(1,keepdims=True).clip(1e-9)
    d=series[-1].copy(); markov=[]
    for _ in range(pr): d=d@P; markov.append(d.copy())
    markov=np.array(markov)
    # LSTM forecast (noise-augmented so the short series is trainable)
    if progress: progress(0.9,desc="Training forecast LSTM…")
    Wn=min(3,max(T-1,1)); torch.manual_seed(0); np.random.seed(0); Xtr,Ytr=[],[]
    for _ in range(80):
        s=np.clip(series+np.random.normal(0,0.01,series.shape),0,1)
        for t in range(Wn,T): Xtr.append(s[t-Wn:t]); Ytr.append(s[t])
    net=ForecastLSTM().to(device); opt=torch.optim.Adam(net.parameters(),lr=5e-3)
    Xtr=torch.tensor(np.array(Xtr)).float(); Ytr=torch.tensor(np.array(Ytr)).float()
    for _ in range(400): opt.zero_grad(); F.mse_loss(net(Xtr.to(device)),Ytr.to(device)).backward(); opt.step()
    seq=list(series.copy()); lstm=[]
    with torch.no_grad():
        for _ in range(pr):
            nxt=net(torch.tensor(np.array(seq[-Wn:])[None]).float().to(device)).cpu().numpy()[0]
            nxt=np.clip(nxt,0,None); nxt/=nxt.sum(); lstm.append(nxt); seq.append(nxt)
    lstm=np.array(lstm); fut=[years[-1]+k+1 for k in range(pr)]
    plt.figure(figsize=(13,7))
    for c in range(N):
        plt.plot(years,100*series[:,c],"-o",color=PALETTE[c],lw=2.4,label=CLASSES[c])
        plt.plot([years[-1]]+fut,[100*series[-1,c]]+list(100*lstm[:,c]),"--",color=PALETTE[c],lw=2)
        plt.plot([years[-1]]+fut,[100*series[-1,c]]+list(100*markov[:,c]),":",color=PALETTE[c],lw=2,alpha=.65)
    plt.axvline(years[-1],color="grey",ls="--",alpha=.5); plt.xlabel("year"); plt.ylabel("% of area"); plt.grid(alpha=.3)
    plt.title("Land-cover forecast — solid: observed · dashed: LSTM · dotted: Markov baseline"); plt.legend(ncol=4,fontsize=9)
    g=f"{out}/forecast.png"; plt.tight_layout(); plt.savefig(g,dpi=130); plt.close()
    md=f"**Forecast** · observed {years[0]}–{years[-1]} · predicting {fut}\n\n| Class | {years[-1]} | "+" | ".join(f"{y} LSTM | {y} Markov" for y in fut)+" |\n|"+"---|"*(2+2*pr)+"\n"
    for c in range(N):
        md+=f"| {CLASSES[c]} | {100*series[-1,c]:.1f}% | "+" | ".join(f"{100*lstm[k,c]:.1f}% | {100*markov[k,c]:.1f}%" for k in range(pr))+" |\n"
    md+="\n_**LSTM** = deep learning forecaster. **Markov** = classical baseline. Illustrative 'if current trends continue' projection — not a reliable prediction._"
    return os.path.abspath(g), md
def analyze_forecast(lat,lon,ys,ye,pred,progress=gr.Progress()):
    try: return run_forecast(float(lat),float(lon),int(ys),int(ye),int(pred),progress=progress)
    except Exception:
        import traceback; return None,f"### Error\n```\n{traceback.format_exc()[-1200:]}\n```"

CSS="""
.gradio-container{max-width:100%!important;padding:0 20px 26px!important}
#hdr{position:relative;overflow:hidden;background:#0d1f2d;color:#e9f1f8;padding:26px 30px;border-radius:12px;border:1px solid #16324a}
#hdr::before{content:"";position:absolute;inset:-60%;pointer-events:none;
 background:repeating-radial-gradient(circle at 13% 50%, transparent 0 26px, rgba(90,200,150,.10) 26px 27px)}
#hdr::after{content:"";position:absolute;top:50%;left:13%;width:560px;height:560px;transform:translate(-50%,-50%);
 border-radius:50%;pointer-events:none;animation:sweep 7s linear infinite;
 background:conic-gradient(from 0deg, rgba(80,220,150,.16), transparent 24%)}
@keyframes sweep{to{transform:translate(-50%,-50%) rotate(360deg)}}
#hdr h1{position:relative;z-index:1;margin:0;font-size:23px;font-weight:700;letter-spacing:.2px}
#hdr p{position:relative;z-index:1;margin:8px 0 0;font-size:13.5px;color:#aecbdd;max-width:760px}
.mrow{display:flex;gap:12px;margin:8px 0}
.metric{flex:1;border-radius:10px;padding:13px 15px;background:#f6f8fa;border:1px solid #e3e8ee;border-left:4px solid #8a97a3}
.metric .v{font-size:21px;font-weight:700;color:#1a2a36;line-height:1.1}.metric .l{font-size:12px;color:#5f6f7c;margin-top:4px}
.m1{border-left-color:#c0392b}.m2{border-left-color:#1f78b4}.m3{border-left-color:#2e7d32}.m4{border-left-color:#6a3d9a}
"""
theme=gr.themes.Soft(primary_hue="blue",secondary_hue="green",neutral_hue="slate",font=["Inter","system-ui","sans-serif"])
with gr.Blocks(title="Land cover from Sentinel-1 radar") as demo:
    gr.HTML('<div id="hdr"><h1>Land Cover from Sentinel-1 Radar</h1>'
            '<p>Each month is classified with a U-Net from radar alone; a temporal model keeps the sequence '
            'consistent and separates real change from seasonal noise. Sentinel-1 VV and VH, seven land-cover classes.</p></div>')
    with gr.Tabs():
        with gr.Tab("Monthly land cover"):
            with gr.Group():
                with gr.Row():
                    preset=gr.Dropdown(list(PRESETS.keys()),value=list(PRESETS.keys())[0],label="Region (choose Custom to enter coordinates)",scale=3)
                    year=gr.Number(value=2021,precision=0,label="Year (2016–2024)",scale=1)
                with gr.Row():
                    lat=gr.Number(value=49.4875,label="Latitude (Custom only)"); lon=gr.Number(value=8.4660,label="Longitude (Custom only)")
                    start_month=gr.Slider(1,12,value=1,step=1,label="Start month (first of 12)")
            btn=gr.Button("Run",variant="primary",size="lg")
            metrics=gr.HTML()
            with gr.Tabs():
                with gr.Tab("Maps"):
                    img_monthly=gr.Image(label="12 monthly land-cover maps")
                    img_change=gr.Image(label="Change-detection map")
                    img_conf=gr.Image(label="Confidence map")
                with gr.Tab("Trends and validation"):
                    img_classpct=gr.Image(label="Class area % across 12 months")
                    img_trans=gr.Image(label="Class transition matrix (month 1 to 12)")
                    img_dw=gr.Image(label="Independent check against Google Dynamic World")
                with gr.Tab("Summary"):
                    summary=gr.Markdown()
            btn.click(analyze,[preset,lat,lon,start_month,year],
                      [img_monthly,img_classpct,img_change,img_conf,img_trans,img_dw,summary,metrics])
        with gr.Tab("Hotspot finder"):
            gr.Markdown("Scan an area and rank where land changed most between two years. It compares stable annual "
                        "maps (real change, not single-date noise) at a coarse resolution for a fast scan; the "
                        "command-line hotspot_finder.py runs at full 10 m resolution. Typically 20 to 30 seconds.")
            with gr.Group():
                with gr.Row():
                    hlat=gr.Number(value=48.10,label="Latitude"); hlon=gr.Number(value=11.60,label="Longitude")
                    hsize=gr.Number(value=24,label="Area size (km)"); hgrid=gr.Slider(4,8,value=6,step=1,label="Grid")
                with gr.Row():
                    hya=gr.Number(value=2019,precision=0,label="Year A"); hyb=gr.Number(value=2023,precision=0,label="Year B")
            hbtn=gr.Button("Find hotspots",variant="primary",size="lg")
            hsummary=gr.Markdown()
            himg1=gr.Image(label="Change-intensity grid (top cell marked)")
            himg2=gr.Image(label="Land cover year A vs year B, and change")
            hbtn.click(analyze_hotspot,[hlat,hlon,hsize,hgrid,hya,hyb],[himg1,himg2,hsummary])
        with gr.Tab("Forecast"):
            gr.Markdown("Project land cover a few years ahead from past annual maps. An LSTM forecaster is shown "
                        "alongside a Markov transition-probability baseline. This is an illustrative 'if current "
                        "trends continue' projection, not a reliable prediction. The first run takes about a minute.")
            with gr.Group():
                with gr.Row():
                    flat=gr.Number(value=48.15,label="Latitude"); flon=gr.Number(value=11.58,label="Longitude")
                    fys=gr.Number(value=2018,precision=0,label="From year"); fye=gr.Number(value=2023,precision=0,label="To year")
                    fpred=gr.Slider(1,3,value=2,step=1,label="Years ahead")
            fbtn=gr.Button("Forecast",variant="primary",size="lg")
            fimg=gr.Image(label="Observed and forecast (solid = observed, dashed = LSTM, dotted = Markov)")
            fsummary=gr.Markdown()
            fbtn.click(analyze_forecast,[flat,flon,fys,fye,fpred],[fimg,fsummary])
if __name__=="__main__":
    demo.queue(max_size=8).launch(theme=theme,css=CSS,share=True,server_name="0.0.0.0",allowed_paths=[os.path.abspath(".")])
