#!/usr/bin/env python3
"""P5 (2026-11-24 cutout fix): remove the card-photo backdrop patch left BETWEEN the two shoes of the
8 pairs re-cut in 7c6a44a (n001/n002/n006/n015/n035/n049/n050/n056).
GrabCut on the existing layer: shoe cores (outer columns) = foreground, low-saturation pixels in the
centre strip = backdrop; only backdrop that lies between the shoes (shoe on both sides in that row, or
the middle band) is removed, so soles/outer edges are never touched. Layer size, rect and hotspot unchanged.
Reads the original layers from 326fc1d so re-runs are idempotent. Usage: fix_shoe_gaps.py [--write]"""
import io, json, subprocess, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from cutout_matte import soft_matte, decontaminate
BASE = "326fc1d"
IDS = ["shoes-n001", "shoes-n002", "shoes-n006", "shoes-n015", "shoes-n035", "shoes-n049", "shoes-n050", "shoes-n056"]

def gapcut(A):
    rgb=A[...,:3].copy(); a=(A[...,3]>128).astype(np.uint8); h,w=a.shape
    f=rgb.astype(int); sat=f.max(2)-f.min(2)
    gc=np.full((h,w),cv2.GC_PR_FGD,np.uint8)
    gc[a==0]=cv2.GC_BGD
    core=cv2.erode(a,np.ones((3,3),np.uint8),iterations=10)>0
    xs=np.arange(w)[None,:].repeat(h,0)
    gc[core&((xs<0.3*w)|(xs>0.7*w))]=cv2.GC_FGD
    strip=(a>0)&(np.abs(xs-w/2)<0.03*w)&(sat<45)
    gc[strip]=cv2.GC_BGD
    # fill transparent area RGB with backdrop median so the bg model learns it, not black
    med=np.median(f[strip],0) if strip.sum()>20 else np.array([180,170,160])
    work=rgb.copy(); work[a==0]=med.astype(np.uint8)
    cv2.setRNGSeed(42)
    cv2.grabCut(cv2.cvtColor(work,cv2.COLOR_RGB2BGR),gc,None,np.zeros((1,65)),np.zeros((1,65)),6,cv2.GC_INIT_WITH_MASK)
    m=((gc==1)|(gc==3)).astype(np.uint8)&a
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(m,8)
    order=np.argsort(-st[1:,4])[:2]+1
    keep=np.isin(lab,order)
    # fill small holes
    inv=(~keep).astype(np.uint8); n2,l2,s2,_=cv2.connectedComponentsWithStats(inv,4)
    for j in range(1,n2):
        x,y,ww,hh,ar=s2[j]
        if ar<300 and x>0 and y>0 and x+ww<w and y+hh<h: keep|=(l2==j)
    keep=keep&(a>0)
    # only remove backdrop that lies BETWEEN the two shoes (shoe on both sides in that row) or in the middle band
    kl=np.maximum.accumulate(keep,axis=1); kr=np.maximum.accumulate(keep[:,::-1],axis=1)[:,::-1]
    between=kl&kr
    central=np.abs(xs-w/2)<0.06*w
    removable=(a>0)&~keep&(between|central)
    keep=(a>0)&~removable
    return keep, med

def main():
    write = "--write" in sys.argv
    log = {}
    for iid in IDS:
        A = np.array(Image.open(io.BytesIO(subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE}:assets/zones/layers/{iid}.webp"],
                     capture_output=True, check=True).stdout)).convert("RGBA"))
        keep, med = gapcut(A)
        a = A[..., 3].astype(np.float32) / 255
        removed = (A[..., 3] > 128) & ~keep
        # soft edge only where backdrop was removed; elsewhere the alpha is untouched
        soft = soft_matte(keep, shrink=0.0)
        near = cv2.dilate(removed.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
        alpha = np.where(near, np.minimum(a, soft), a)
        rgb = decontaminate(A[..., :3].copy(), alpha, depth=2, band=2, wall=med)
        rgb = np.where(near[..., None], rgb, A[..., :3])
        out = np.dstack([rgb, np.clip(alpha * 255 + .5, 0, 255).astype(np.uint8)])
        out[out[..., 3] == 0, :3] = 0
        log[iid] = dict(removed_px=int(removed.sum()), backdrop=[int(v) for v in med])
        print(iid, log[iid])
        if write:
            p = ROOT / f"assets/zones/layers/{iid}.webp"
            Image.fromarray(out).save(p, "WEBP", quality=90, method=6)
    if write:
        (ROOT / "data/source/display/shoe-gap-fix.json").write_text(json.dumps(log, indent=1) + "\n")

if __name__ == "__main__":
    main()
