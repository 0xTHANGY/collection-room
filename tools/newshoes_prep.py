#!/usr/bin/env python3
"""Source preparation for the V2.4.0 new-shoes rooms (pilot: zone-01 = shoes-c).

Deterministic and local: no AI redrawing. Card cutouts keep the original photo pixels (only the
background and the catalogue label are removed); hotspot boxes/masks come from the zone painting.
Reads the extracted delivery (work/newshoes/new-shoes-final, read-only) and writes the second source
directory data/source/newshoes/ (committed) plus a review report. The normal build never runs this.
Usage: python3 tools/newshoes_prep.py [cutouts|zone]
"""
import json, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data/source/newshoes'
DELIVERY=ROOT/'work/newshoes/new-shoes-final'
PILOT=ROOT/'work/newshoes/pilot'

EDGE_T=12.0
def lab_of(rgb): return cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)

def find_label(rgb):
    """Catalogue label = thin dark vertical rule + text to its right, lower-right corner.
    Returns (x0,y0) of the label rectangle (rule x-6, first text row-10) or None."""
    h,w=rgb.shape[:2]
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    x0=int(w*.55);y0=int(h*.72)
    sub=gray[y0:,x0:]
    dark=(sub<150).astype(np.uint8)
    # vertical rule: a column run of dark pixels, 1-4 px wide, >= 2.5% of height
    col=cv2.morphologyEx(dark,cv2.MORPH_OPEN,np.ones((int(h*.025),1),np.uint8))
    n,l,s,_=cv2.connectedComponentsWithStats(col)
    best=None
    for i in range(1,n):
        x,y,cw,ch,a=s[i]
        if cw<=6 and ch>=h*.025 and (best is None or x>best[0]): best=(x,y,cw,ch)
    return None if best is None else (x0+best[0],y0+best[1],best[2],best[3])

def cutout(path):
    rgb=np.array(Image.open(path).convert('RGB'));h,w=rgb.shape[:2]
    lab=lab_of(rgb)
    rule=find_label(rgb)
    label=np.zeros((h,w),bool)
    if rule:
        x,y,cw,ch=rule
        label[max(0,y-int(h*.03)):,max(0,x-8):]=True   # rule + all text (right of / below the rule)
    # Background = everything reachable from the photo border without crossing an edge. Cast/contact shadows
    # are smooth gradients, so they flood into the background; object outlines (and the edges of laces,
    # stitching, insoles) are strong gradients and wall the object in. No colour model, no AI.
    soft=cv2.GaussianBlur(lab,(0,0),1.3)
    gx=cv2.Sobel(soft,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(soft,cv2.CV_32F,0,1,ksize=3)
    grad=np.sqrt((gx**2+gy**2)).max(2)
    edge=(grad>EDGE_T)
    edge=cv2.dilate(np.uint8(edge)*255,np.ones((3,3),np.uint8))>0
    free=(~edge)&(~label)
    n,fl,fs,_=cv2.connectedComponentsWithStats(np.uint8(free),connectivity=4)
    border_ids=set(np.unique(np.concatenate([fl[0],fl[-1],fl[:,0],fl[:,-1]])))-{0}
    bgmask=np.isin(fl,list(border_ids))
    # bring the 1-2 px edge band that belongs to the background back: grow bg into edge pixels that are
    # not object-dark (keeps antialiased outline, avoids a halo)
    m=np.uint8(~bgmask)*255
    m[label]=0
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    dist=np.zeros((h,w),np.float32);model=lab
    n,l,s,_=cv2.connectedComponentsWithStats(m)
    if n>1:
        big=s[1:,4].max()
        m=np.uint8(np.isin(l,[k for k in range(1,n) if s[k,4]>big*.08]))*255
    # fill enclosed holes (insoles, tongue gaps) except large openings
    inv=255-m;n,hl,hs,_=cv2.connectedComponentsWithStats(inv)
    for k in range(1,n):
        x,y,cw,ch,a=hs[k]
        if x==0 or y==0 or x+cw==w or y+ch==h: continue
        if a<w*h*.004: m[hl==k]=255
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    # peel the soft-shadow halo that is still inside the flood boundary: grabCut on a narrow band only
    sure_fg=cv2.erode(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
    gc=np.full((h,w),cv2.GC_PR_FGD,np.uint8)
    gc[cv2.dilate(m,np.ones((9,9),np.uint8))==0]=cv2.GC_BGD
    gc[sure_fg>0]=cv2.GC_FGD
    gc[label]=cv2.GC_BGD
    cv2.setRNGSeed(7)
    cv2.grabCut(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),gc,None,np.zeros((1,65)),np.zeros((1,65)),3,cv2.GC_INIT_WITH_MASK)
    m=np.uint8((gc==1)|(gc==3))*255
    n,l,s,_=cv2.connectedComponentsWithStats(m)
    if n>1:
        m=np.uint8(np.isin(l,[k for k in range(1,n) if s[k,4]>s[1:,4].max()*.08]))*255
    inv=255-m;n,hl,hs,_=cv2.connectedComponentsWithStats(inv)
    for k in range(1,n):
        x,y,cw,ch,a=hs[k]
        if x==0 or y==0 or x+cw==w or y+ch==h: continue
        if a<w*h*.004: m[hl==k]=255
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    m=cv2.erode(m,np.ones((3,3),np.uint8))              # drop the 1px light halo
    alpha=cv2.GaussianBlur(m,(0,0),.8)
    info=dict(size=[w,h],label_rule=None if not rule else [int(v) for v in rule],
              area=round(float((m>0).mean()),4))
    return rgb,alpha,m,label,info

def gate(m,label,info,rgb):
    h,w=m.shape;reasons=[]
    n,l,s,_=cv2.connectedComponentsWithStats((m>0).astype(np.uint8))
    comps=[s[k] for k in range(1,n) if s[k,4]>h*w*.002]
    info['components']=len(comps)
    if not 1<=len(comps)<=2: reasons.append(f'components={len(comps)} (expected 1-2 shoes)')
    ys,xs=np.where(m>0)
    if not len(xs): return ['empty']
    if xs.min()<3 or ys.min()<3 or xs.max()>w-4 or ys.max()>h-4: reasons.append('touches photo edge')
    if (m[label]>0).any(): reasons.append('overlaps label')
    if not .08<(m>0).mean()<.55: reasons.append('area implausible')
    # label text must be gone: no foreground pixel inside the rule/text rectangle
    # halo: ring just outside the mask must be close to background brightness
    ring=(cv2.dilate(m,np.ones((7,7),np.uint8))>0)&~(cv2.dilate(m,np.ones((2,2),np.uint8))>0)
    if ring.any():
        far=~(cv2.dilate(m,np.ones((41,41),np.uint8))>0)&~label
        diff=abs(rgb[ring].astype(float).mean()-rgb[far].astype(float).mean())
        info['edge_vs_bg_lum']=round(float(diff),1)
    return reasons

def run_cutouts():
    items=json.loads((SRC/'items.json').read_text())['items']
    (SRC/'cutouts').mkdir(exist_ok=True);PILOT.mkdir(parents=True,exist_ok=True)
    report={};tiles=[]
    for it in items:
        rgb,alpha,m,label,info=cutout(DELIVERY/'originals'/it['source_file'])
        reasons=gate(m,label,info,rgb);info['pass']=not reasons;info['reasons']=reasons
        ys,xs=np.where(m>0);pad=int(max(xs.max()-xs.min(),ys.max()-ys.min())*.04)
        x0,x1=max(0,xs.min()-pad),min(rgb.shape[1],xs.max()+pad+1);y0,y1=max(0,ys.min()-pad),min(rgb.shape[0],ys.max()+pad+1)
        rgba=np.dstack([rgb,alpha])[y0:y1,x0:x1].copy()
        rgba[rgba[:,:,3]==0,:3]=0
        Image.fromarray(rgba).save(SRC/'cutouts'/f"{it['id']}.webp",'WEBP',lossless=True,method=6,exact=False)
        report[it['id']]=info
        t=Image.fromarray(rgba);t.thumbnail((300,300))
        chk=Image.fromarray(np.full((310,310,3),(70,100,130),np.uint8))
        chk.paste(t,(5,5),t);tiles.append((it['id'],chk,info))
    sheet=Image.new('RGB',(3*320,((len(tiles)+2)//3)*350),'white')
    from PIL import ImageDraw
    d=ImageDraw.Draw(sheet)
    for k,(iid,chk,info) in enumerate(tiles):
        x=(k%3)*320;y=(k//3)*350;sheet.paste(chk,(x+5,y));d.text((x+8,y+312),f"{iid} {'PASS' if info['pass'] else 'FAIL'} {'; '.join(info['reasons'])}"[:60],fill='black')
    sheet.save(PILOT/'cutout-contact-sheet.jpg',quality=90)
    (PILOT/'cutout-report.json').write_text(json.dumps(report,indent=1))
    print(json.dumps({k:(v['pass'],v['reasons']) for k,v in report.items()}))

# approximate pair rectangles in the 1536x1024 zone painting (visual estimate, +-15 px; GrabCut refines)
ZONE_C={'shoes-n010':(320,316,560,592),'shoes-n011':(660,246,900,584),'shoes-n012':(1024,316,1282,598),
        'shoes-n007':(222,614,498,962),'shoes-n008':(614,578,920,972),'shoes-n009':(1042,624,1342,962)}
IDENT_T=5.0
CLOSE_K=45
HOLE_MAX=40000
MARGIN=19
DIFF_T=44.0
WEAK_T=12.0
def run_zone(zid='shoes-c',zone_png='zones/new-shoes-zone-01.png',plate_png=None):
    painting=Image.open(DELIVERY/zone_png).convert('RGB')
    (SRC/'zones').mkdir(exist_ok=True);(SRC/'masks').mkdir(exist_ok=True)
    painting.save(SRC/'zones'/f'{zid}.webp','WEBP',lossless=True,method=6)       # fresh pixel-only image: no metadata
    if plate_png:
        g=ROOT/'data/source/generated';Image.open(plate_png).convert('RGB').save(g/'shoes-zone-c-rug-v240.webp','WEBP',lossless=True,method=6)
    rgb=np.array(painting);h,w=rgb.shape[:2]
    # The generated empty-rug plate is the reference: |painting - plate| (after a rigid-ish ECC alignment on the
    # untouched wall) is large on the shoes, small on the rug, intermediate on the soft cast shadows.
    plate=np.array(Image.open(SRC.parent/'generated/shoes-zone-c-rug-v240.webp').convert('RGB').resize((w,h)))
    g0=cv2.GaussianBlur(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32),(0,0),1.5)
    g1=cv2.GaussianBlur(cv2.cvtColor(plate,cv2.COLOR_RGB2GRAY).astype(np.float32),(0,0),1.5)
    top=np.zeros((h,w),np.uint8);top[:300]=255;warp=np.eye(2,3,dtype=np.float32)
    _,warp=cv2.findTransformECC(g0,g1,warp,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,200,1e-6),top,5)
    plate=cv2.warpAffine(plate,warp,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT)
    la=cv2.cvtColor(cv2.GaussianBlur(rgb,(0,0),2),cv2.COLOR_RGB2LAB).astype(np.float32)
    lb=cv2.cvtColor(cv2.GaussianBlur(plate,(0,0),2),cv2.COLOR_RGB2LAB).astype(np.float32)
    diff=np.linalg.norm(la-lb,axis=2)
    # Cast/contact shadows are the plate multiplied by a neutral factor (darker, same hue); objects are not.
    pa=cv2.GaussianBlur(rgb.astype(np.float32),(0,0),5);pb=cv2.GaussianBlur(plate.astype(np.float32),(0,0),5)
    ratio=(pa+1)/(pb+1);rm=ratio.mean(2);rp=np.ptp(ratio,axis=2)
    shadow=(rm>.45)&(rm<.97)&(rp<.07)
    strong=(diff>DIFF_T)&~shadow
    near=cv2.dilate(np.uint8(strong)*255,np.ones((25,25),np.uint8))>0
    weak=(diff>WEAK_T)&(rm>1.0)&~shadow&near            # pale parts brighter than the rug (laces, cream leather)
    weak=cv2.morphologyEx(np.uint8(weak)*255,cv2.MORPH_OPEN,np.ones((7,7),np.uint8))>0
    fgmap=strong|weak
    items={}
    ov=rgb.copy()
    for iid,(x0,y0,x1,y1) in ZONE_C.items():
        roi=np.zeros((h,w),bool);roi[max(0,y0-12):y1+12,max(0,x0-12):x1+12]=True
        m=np.uint8(roi&fgmap)*255
        m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
        m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
        n,l,st,_=cv2.connectedComponentsWithStats(m)
        keep=[k for k in range(1,n) if st[k,4]>st[1:,4].max()*.15]
        m=np.uint8(np.isin(l,keep))*255
        m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(CLOSE_K,CLOSE_K)))
        inv=255-m;n,hl,hs,_=cv2.connectedComponentsWithStats(inv)
        for k in range(1,n):
            x,y,cw,ch,a=hs[k]
            if x==0 or y==0 or x+cw==w or y+ch==h: continue
            if a<HOLE_MAX: m[hl==k]=255
        # pixels identical to the empty plate (wall, floor between the boots) cannot be shoe: drop them from the closed mask
        m[diff<IDENT_T]=0
        m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
        m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(MARGIN,MARGIN)))   # margin: never leave shoe pixels outside the layer
        ys,xs=np.where(m>0)
        items[iid]=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
        Image.fromarray(m).save(SRC/'masks'/f'{iid}.png',optimize=True)
        c,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);cv2.drawContours(ov,c,-1,(0,255,0),2)
        cv2.rectangle(ov,tuple(items[iid][:2]),tuple(items[iid][2:]),(255,0,255),1)
    layout={'zone':zid,'image_size':[w,h],'items':{k:{'bbox':v,'confidence':'local GrabCut on the zone painting, visually checked','note':'V2.4.0 pilot, mask in data/source/newshoes/masks'} for k,v in items.items()}}
    (ROOT/'data/source/layout'/f'{zid}.json').write_text(json.dumps(layout,indent=1)+'\n')
    Image.fromarray(ov).save(PILOT/'zone-masks-overlay.jpg',quality=90)
    print(json.dumps(items))

if __name__=='__main__':
    cmd=sys.argv[1] if len(sys.argv)>1 else 'cutouts'
    if cmd=='cutouts':run_cutouts()
    elif cmd=='zone':run_zone(plate_png=sys.argv[2] if len(sys.argv)>2 else None)
