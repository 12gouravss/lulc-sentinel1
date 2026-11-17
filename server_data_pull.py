import argparse, os
import numpy as np
import ee, geemap

PROJECT = "lulc-sentinel-513912"

REGIONS_TRAIN = {
    "po_valley":    [9.00, 45.15, 9.25, 45.35],
    "rhine_neckar": [8.40, 49.40, 8.65, 49.58],
    "flevoland":    [5.35, 52.45, 5.65, 52.65],
    "andalusia":    [-6.05, 37.30, -5.80, 37.50],
    "danube":       [18.90, 47.30, 19.15, 47.50],
    "saxony":       [12.20, 51.25, 12.50, 51.45],
    "alps":         [10.75, 46.80, 11.00, 47.00],   # snow/ice, bare, shrubland, trees
    "greece_scrub": [22.00, 39.00, 22.25, 39.20],   # Mediterranean shrubland
    "camargue":     [4.40, 43.40, 4.65, 43.58],     # coastal water + wetland
}
REGIONS_TEST = {
    "beauce":    [1.70, 48.00, 1.95, 48.20],
    "catalonia": [1.90, 41.45, 2.20, 41.65],
    "provence":  [5.80, 43.55, 6.05, 43.75],         # Mediterranean shrub/built (held out)
}
YEAR = 2021
MONTHS = list(range(1, 13))
SCALE = 10
PATCH = 256
TILE_DIR = "tiles"


def get_s1_composite(geom, year, month):
    start = ee.Date.fromYMD(year, month, 1); end = start.advance(1, "month")
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(geom).filterDate(start, end)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
          .select(["VV", "VH"]))
    comp = s1.median().focalMedian(1.5, "circle", "pixels")
    vv = comp.select("VV").clamp(-25, 0).unitScale(-25, 0)
    vh = comp.select("VH").clamp(-30, -5).unitScale(-30, -5)
    return vv.addBands(vh).rename(["VV", "VH"]).toFloat()


def get_labels(geom):
    # 7 classes: 0 Water 1 Trees 2 Grassland 3 Cropland 4 Built-up 5 Bare/Sparse 6 Wetland
    # (Shrubland→Grassland, Snow→Bare, Mangrove→Water, Moss→Grassland — kept at 7 because
    #  SAR cannot reliably separate finer classes; justified in the report.)
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().clip(geom)
    frm = [80, 10, 20, 30, 40, 50, 60, 90, 95, 70, 100]
    to  = [ 0,  1,  2,  2,  3,  4,  5,  6,  0,  5,   2]
    return wc.remap(frm, to).rename("label").toByte()


def download_tile(name, box, year, month, out_dir):
    geom = ee.Geometry.Rectangle(box)
    stack = get_s1_composite(geom, year, month).addBands(get_labels(geom))
    path = os.path.join(out_dir, f"{name}_{year}_{month:02d}.tif")
    if os.path.exists(path):
        print(f"  exists: {path}"); return path
    geemap.download_ee_image(stack, path, region=geom, scale=SCALE, crs="EPSG:4326")
    print(f"  saved:  {path}")
    return path


def patchify(tif_path, patch=PATCH, stride=PATCH):
    import rasterio
    with rasterio.open(tif_path) as src:
        arr = src.read()
    img = np.stack([arr[0], arr[1]], -1); lab = arr[2]
    H, W = lab.shape; X, Y = [], []
    for i in range(0, H - patch + 1, stride):
        for j in range(0, W - patch + 1, stride):
            xp = img[i:i+patch, j:j+patch]; yp = lab[i:i+patch, j:j+patch]
            # skip patches with missing image OR missing label (clean data only)
            if np.isnan(xp).mean() < 0.1 and not np.isnan(yp).any():
                X.append(np.nan_to_num(xp, nan=0.0).astype(np.float32))
                Y.append(np.clip(yp, 0, 6).astype(np.uint8))   # 7 classes (0..6)
    return X, Y


def build(regions, months, out_dir):
    X, Y = [], []
    for name, box in regions.items():
        for m in months:
            p = download_tile(name, box, YEAR, m, out_dir)
            xs, ys = patchify(p)
            X += xs; Y += ys
            print(f"    {name} {m:02d}: {len(xs)} patches (running total {len(X)})", flush=True)
    return np.array(X), np.array(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    ee.Initialize(project=PROJECT)
    print("Earth Engine initialized on project", PROJECT)
    os.makedirs(TILE_DIR, exist_ok=True)
    if args.test:
        tr = {"rhine_neckar": REGIONS_TRAIN["rhine_neckar"]}
        te = {"catalonia": REGIONS_TEST["catalonia"]}
        months = [6]; print(">>> TEST MODE: 1+1 regions, June")
    else:
        tr, te, months = REGIONS_TRAIN, REGIONS_TEST, MONTHS
        print(">>> FULL MODE: 6 train + 2 test regions x 12 months")
    print("\n== TRAIN =="); Xtr, Ytr = build(tr, months, TILE_DIR)
    print("\n== TEST ==");  Xte, Yte = build(te, months, TILE_DIR)
    np.savez_compressed("train.npz", X=Xtr, Y=Ytr)
    np.savez_compressed("test.npz",  X=Xte, Y=Yte)
    print(f"\nSaved train.npz {Xtr.shape} and test.npz {Xte.shape}")
    uniq, cnt = np.unique(Ytr, return_counts=True)
    print("Train class pixel counts:", dict(zip(uniq.tolist(), cnt.tolist())))


if __name__ == "__main__":
    main()
