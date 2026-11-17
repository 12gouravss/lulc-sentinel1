import os, numpy as np

passed = failed = 0
def ok(label, cond, extra=""):
    global passed, failed
    print(f"  [{'OK  ' if cond else 'FAIL'}] {label}" + (f"   {extra}" if extra else ""))
    if cond: passed += 1
    else: failed += 1
    return cond

print("== 1. imports ==")
for m in ["torch","numpy","ee","geemap","geedim","rasterio","gradio","matplotlib"]:
    try: __import__(m); ok(m, True)
    except Exception as e: ok(m, False, str(e)[:55])
import torch, torch.nn as nn
print(f"        torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")

print("\n== 2. files present ==")
for f in ["server_data_pull.py","train.py","train_multitemporal.py","lstm_temporal.py",
          "prototype.py","app.py","hotspot_finder.py","train.npz","test.npz",
          "unet_s1_lulc.pth","unet_vv_only.pth","unet_multitemporal.pth","lstm_temporal.pth"]:
    ok(f, os.path.exists(f))
ok("tiles/ directory", os.path.isdir("tiles"))
ok("outputs/figures/", os.path.isdir("outputs/figures"))

# model classes (two U-Net naming conventions + the LSTM) 
class DCa(nn.Module):   # train.py convention (.net)
    def __init__(s,a,b):
        super().__init__(); s.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(True),
                                                nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(True))
    def forward(s,x): return s.net(x)
class UNetA(nn.Module):
    def __init__(s,in_ch=2,n=7,base=64):
        super().__init__()
        s.d1=DCa(in_ch,base);s.d2=DCa(base,base*2);s.d3=DCa(base*2,base*4);s.d4=DCa(base*4,base*8)
        s.pool=nn.MaxPool2d(2);s.bott=DCa(base*8,base*16)
        s.up4=nn.ConvTranspose2d(base*16,base*8,2,2);s.u4=DCa(base*16,base*8)
        s.up3=nn.ConvTranspose2d(base*8,base*4,2,2);s.u3=DCa(base*8,base*4)
        s.up2=nn.ConvTranspose2d(base*4,base*2,2,2);s.u2=DCa(base*4,base*2)
        s.up1=nn.ConvTranspose2d(base*2,base,2,2);s.u1=DCa(base*2,base)
        s.out=nn.Conv2d(base,n,1)
    def forward(s,x):
        c1=s.d1(x);c2=s.d2(s.pool(c1));c3=s.d3(s.pool(c2));c4=s.d4(s.pool(c3));b=s.bott(s.pool(c4))
        x=s.u4(torch.cat([s.up4(b),c4],1));x=s.u3(torch.cat([s.up3(x),c3],1))
        x=s.u2(torch.cat([s.up2(x),c2],1));x=s.u1(torch.cat([s.up1(x),c1],1));return s.out(x)
class DCb(nn.Module):   # train_multitemporal.py convention (.n)
    def __init__(s,a,b):
        super().__init__(); s.n=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(True),
                                              nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(True))
    def forward(s,x): return s.n(x)
class UNetB(nn.Module):
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

print("\n== 3. weights load + forward pass ==")
def check_unet(cls, path, in_ch):
    try:
        m=cls(in_ch=in_ch); m.load_state_dict(torch.load(path,map_location="cpu")); m.eval()
        with torch.no_grad(): y=m(torch.randn(1,in_ch,256,256))
        ok(f"{path}", tuple(y.shape)==(1,7,256,256), f"out {tuple(y.shape)}")
    except Exception as e: ok(f"{path}", False, str(e)[:70])
check_unet(UNetA,"unet_s1_lulc.pth",2)
check_unet(UNetA,"unet_vv_only.pth",1)
check_unet(UNetB,"unet_multitemporal.pth",10)
try:
    l=TemporalLSTM(); l.load_state_dict(torch.load("lstm_temporal.pth",map_location="cpu")); l.eval()
    with torch.no_grad(): y=l(torch.randn(4,12,7))
    ok("lstm_temporal.pth", tuple(y.shape)==(4,12,7), f"out {tuple(y.shape)}")
except Exception as e: ok("lstm_temporal.pth", False, str(e)[:70])

print("\n== 4. dataset shapes ==")
try:
    d=np.load("train.npz"); ok("train.npz", d["X"].shape[1:]==(256,256,2), str(d["X"].shape))
    t=np.load("test.npz");  ok("test.npz",  t["X"].shape[1:]==(256,256,2), str(t["X"].shape))
except Exception as e: ok("npz load", False, str(e)[:70])

print("\n== 5. Earth Engine (needs auth; only prototype/app/hotspot use it) ==")
try:
    import ee; ee.Initialize(project="lulc-sentinel-513912")
    ok("EE initialize", ee.Number(1).add(1).getInfo()==2)
except Exception as e:
    ok("EE initialize", False, "run: earthengine authenticate --auth_mode=notebook")

print(f"\n {passed} OK, {failed} FAIL ")
print("All training/inference checks green = the project is intact and runnable.")
if failed: print("(EE FAIL is fine if you haven't re-authenticated this session — training/inference still work.)")
