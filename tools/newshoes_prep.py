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

def cutout(path,inset=0):
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
    if inset:   # photos with a framed margin: the frame line walls the border off, so also seed the flood from a ring inside the frame
        border_ids|=set(np.unique(np.concatenate([fl[inset],fl[h-1-inset],fl[:,inset],fl[:,w-1-inset]])))-{0}
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

def run_cutouts(zones=None,outdir=None):
    items=json.loads((SRC/'items.json').read_text())['items']
    if zones: items=[i for i in items if i['zone'] in zones]
    outdir=Path(outdir) if outdir else PILOT
    outdir.mkdir(parents=True,exist_ok=True)
    tag='-'.join(z.split('-')[-1] for z in zones) if zones else 'all'
    (SRC/'cutouts').mkdir(exist_ok=True);PILOT.mkdir(parents=True,exist_ok=True)
    report={};tiles=[]
    for it in items:
        rgb,alpha,m,label,info=cutout(DELIVERY/'originals'/it['source_file'])
        reasons=gate(m,label,info,rgb)
        if reasons:   # retry once with the flood also seeded inside a framed photo margin
            r2=cutout(DELIVERY/'originals'/it['source_file'],inset=48);rs2=gate(r2[2],r2[3],r2[4],r2[0])
            if not rs2: rgb,alpha,m,label,info=r2;reasons=rs2;info['retry']='inset-seed'
        info['pass']=not reasons;info['reasons']=reasons
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
    sheet.save(outdir/(f'cutout-contact-sheet-{tag}.jpg' if zones else 'cutout-contact-sheet.jpg'),quality=90)
    (outdir/(f'cutout-report-{tag}.json' if zones else 'cutout-report.json')).write_text(json.dumps(report,indent=1))
    print(json.dumps({k:(v['pass'],v['reasons']) for k,v in report.items()}))

# approximate pair rectangles in the 1536x1024 zone painting (visual estimate, +-15 px; GrabCut refines)
ZONE_C={'shoes-n010':(320,316,560,592),'shoes-n011':(660,246,900,584),'shoes-n012':(1024,316,1282,598),
        'shoes-n007':(222,614,498,962),'shoes-n008':(614,578,920,972),'shoes-n009':(1042,624,1342,962)}
IDENT_T=5.0
CLOSE_K=45
CLUSTER_K=61
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


ZONE_FILES={'b':'zones/new-shoes-zone-test.png','d':'zones/new-shoes-zone-02.png','e':'zones/new-shoes-zone-03-v2.png','f':'zones/new-shoes-zone-04-v2.png','g':'zones/new-shoes-zone-05.png','h':'zones/new-shoes-zone-06-v2.png','i':'zones/new-shoes-zone-07.png','j':'zones/new-shoes-zone-08.png','k':'zones/new-shoes-zone-09.png'}
ZONE_SPLIT={'e':606,'g':618,'h':((382,610),(1132,640)),'i':592,'j':624,'b':634}      # optional: y that separates the back row from the front row where the rows touch
ZONE_NOSHADOW={'f':[(240,190,560,330)]}   # optional rects where dark shoe parts look like shadow: no shadow rule there
ZONE_SEEDS={'d':[(368,795),(690,795),(1170,780),(428,495),(772,480),(1118,480)],
 'e':[(278,818),(735,818),(1102,818),(375,465),(795,480),(1207,495)],
 'f':[(278,765),(765,765),(1192,772),(398,412),(765,382),(1155,375)],
 'g':[(382,795),(765,818),(1162,810),(408,472),(802,488),(1132,412)],
 'h':[(382,810),(772,818),(1170,818),(420,472),(765,480),(1132,495)],
 'i':[(342,780),(765,780),(1192,788),(402,435),(765,450),(1155,435)],
 'j':[(345,810),(742,818),(1162,818),(428,502),(788,488),(1118,480)],
 'k':[(300,788),(772,818),(1185,810),(502,518),(922,518)],
 'b':[(330,832),(750,832),(1185,832),(390,472),(780,495),(1162,495)]}
BODY_T=float(__import__('os').environ.get('NS_BODY',16))
ZONE_PARAMS={'d':(27,19),'e':(27,23),'f':(45,23),'g':(15,23),'h':(45,19),'i':(27,19)}   # letter -> (closing, margin) chosen per zone by work/newshoes/rollout/tune.py (default 33, 19)
def run_zone_auto(letter,plate_png,outdir=None):
    """Other zones (V2.4.0 rollout): same recipe as run_zone (painting vs empty plate), but the pairs are found
    automatically (connected clusters of the difference map) and assigned in delivery order: the first three items
    are the front row (lowest in the picture) left to right, the rest the back row left to right."""
    zid=f'shoes-{letter}';outdir=Path(outdir) if outdir else PILOT;outdir.mkdir(parents=True,exist_ok=True)
    painting=Image.open(DELIVERY/ZONE_FILES[letter]).convert('RGB')
    (SRC/'zones').mkdir(exist_ok=True);(SRC/'masks').mkdir(exist_ok=True)
    painting.save(SRC/'zones'/f'{zid}.webp','WEBP',lossless=True,method=6)
    Image.open(plate_png).convert('RGB').save(ROOT/'data/source/generated'/f'shoes-zone-{letter}-rug-v240.webp','WEBP',lossless=True,method=6)
    import os
    auto_close,auto_margin=ZONE_PARAMS.get(letter,(33,19))
    auto_close=int(os.environ.get('NS_CLOSE',auto_close));auto_margin=int(os.environ.get('NS_MARGIN',auto_margin))
    ids=[i['id'] for i in json.loads((SRC/'items.json').read_text())['items'] if i['zone']==zid]
    rgb=np.array(painting);h,w=rgb.shape[:2]
    plate=np.array(Image.open(ROOT/'data/source/generated'/f'shoes-zone-{letter}-rug-v240.webp').convert('RGB').resize((w,h)))
    g0=cv2.GaussianBlur(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32),(0,0),1.5)
    g1=cv2.GaussianBlur(cv2.cvtColor(plate,cv2.COLOR_RGB2GRAY).astype(np.float32),(0,0),1.5)
    top=np.zeros((h,w),np.uint8);top[:300]=255;warp=np.eye(2,3,dtype=np.float32)
    _,warp=cv2.findTransformECC(g0,g1,warp,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,200,1e-6),top,5)
    plate=cv2.warpAffine(plate,warp,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT)
    la=cv2.cvtColor(cv2.GaussianBlur(rgb,(0,0),2),cv2.COLOR_RGB2LAB).astype(np.float32)
    lb=cv2.cvtColor(cv2.GaussianBlur(plate,(0,0),2),cv2.COLOR_RGB2LAB).astype(np.float32)
    diff=np.linalg.norm(la-lb,axis=2)
    pa=cv2.GaussianBlur(rgb.astype(np.float32),(0,0),5);pb=cv2.GaussianBlur(plate.astype(np.float32),(0,0),5)
    ratio=(pa+1)/(pb+1);rm=ratio.mean(2);rp=np.ptp(ratio,axis=2)
    shadow=(rm>.45)&(rm<.97)&(rp<.07)
    rr=((rgb.astype(np.float32)+1)/(plate.astype(np.float32)+1)).mean(2);shadow&=rr>.42      # blurred shoe edges look like shadow: a pixel this dark is the shoe (or its contact shadow), not cast shadow
    for (rx0,ry0,rx1,ry1) in ZONE_NOSHADOW.get(letter,[]): shadow[ry0:ry1,rx0:rx1]=False
    strong=(diff>DIFF_T)&~shadow
    near=cv2.dilate(np.uint8(strong)*255,np.ones((25,25),np.uint8))>0
    weak=(diff>WEAK_T)&(rm>1.0)&~shadow&near
    weak=cv2.morphologyEx(np.uint8(weak)*255,cv2.MORPH_OPEN,np.ones((7,7),np.uint8))>0
    near2=cv2.dilate(np.uint8(strong)*255,np.ones((61,61),np.uint8))>0     # light/cream shoes on a similar rug: coloured (non-shadow) difference near the pair
    body=(diff>BODY_T)&(rp>=.07)&~shadow&near2
    body=cv2.morphologyEx(np.uint8(body)*255,cv2.MORPH_OPEN,np.ones((7,7),np.uint8))>0
    fgmap=strong|weak|body
    f8=cv2.morphologyEx(np.uint8(fgmap)*255,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))   # thin dark collars/piping must survive
    seeds=ZONE_SEEDS[letter]                               # one (x,y) centre per item, in delivery order (front row L->R, then back row L->R)
    assert len(seeds)==len(ids)
    clus=cv2.morphologyEx(f8,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(CLUSTER_K,CLUSTER_K)))
    yy,xx=np.mgrid[:h,:w];dd=np.stack([np.hypot(xx-sx,(yy-sy)*2.0) for sx,sy in seeds])   # nearest pair centre
    if letter in ZONE_SPLIT:
        nf=3;sp=ZONE_SPLIT[letter];dd=dd.copy()
        if not isinstance(sp,int): (sx0,sy0),(sx1,sy1)=sp;sp=np.round(sy0+(sy1-sy0)*(np.arange(w)-sx0)/(sx1-sx0)).astype(int)[None,:];sp=np.broadcast_to(sp,(h,w))
        else: sp=np.full((h,w),sp)
        front=yy>=sp;dd[:nf][~front[None].repeat(nf,0)]=1e9;dd[nf:][front[None].repeat(len(seeds)-nf,0)]=1e9
    ws=(dd.argmin(0)+1).astype(np.int32);ws[clus==0]=0
    n=len(ids)+1;l=ws;order=list(range(1,n));info={'clusters':len(ids),'expected':len(ids)}
    if any(((l==k)&(clus>0)).sum()<9000 for k in order): raise SystemExit(f'{zid}: a seed region is empty')
    items={};ov=rgb.copy()
    # label every fragment of the difference map by majority vote of the nearest-centre labels (a fragment never straddles two pairs)
    fc=cv2.morphologyEx(f8,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
    nc,cl=cv2.connectedComponents(fc);lv=ws.copy()
    for q in range(1,nc):
        pix=cl==q
        if pix.sum()>6000: continue
        v=np.bincount(ws[pix],minlength=len(ids)+1);v[0]=0
        if v.sum()>0: lv[pix]=v.argmax()
    base={}
    for iid,k in zip(ids,order):
        m=np.uint8((lv==k)&fgmap)*255
        m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
        nn,ll,ss,_=cv2.connectedComponentsWithStats(m)
        big=[q for q in range(1,nn) if ss[q,4]>max(900,ss[1:,4].max()*.05)]
        near_big=cv2.dilate(np.uint8(np.isin(ll,big))*255,np.ones((41,41),np.uint8))>0
        keep=[q for q in range(1,nn) if q in big or (near_big&(ll==q)).any()]   # thin soles/rims are separate pieces: keep what touches the pair, drop far specks
        base[iid]=np.uint8(np.isin(ll,keep))*255
    for iid,k in zip(ids,order):
        m=cv2.morphologyEx(base[iid],cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(auto_close,auto_close)))
        inv=255-m;nn,hl,hs,_=cv2.connectedComponentsWithStats(inv)
        for q in range(1,nn):
            x,y,cw,ch,a=hs[q]
            if x==0 or y==0 or x+cw==w or y+ch==h: continue
            if a<HOLE_MAX: m[hl==q]=255
        others=np.zeros_like(m)
        for j in ids:
            if j!=iid: others|=base[j]
        others=cv2.dilate(others,np.ones((9,9),np.uint8))
        m[(others>0)&(base[iid]==0)]=0                       # never reach into a neighbouring pair
        m[diff<IDENT_T]=0
        m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
        m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(auto_margin,auto_margin)))
        ys,xs=np.where(m>0)
        items[iid]=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
        Image.fromarray(m).save(SRC/'masks'/f'{iid}.png',optimize=True)
        c,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);cv2.drawContours(ov,c,-1,(0,255,0),2)
        cv2.rectangle(ov,tuple(items[iid][:2]),tuple(items[iid][2:]),(255,0,255),1)
        cv2.putText(ov,iid[-3:],(items[iid][0]+6,items[iid][1]+28),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    layout={'zone':zid,'image_size':[w,h],'items':{k:{'bbox':v,'confidence':'local, painting vs generated empty plate, visually checked','note':'V2.4.0 rollout, mask in data/source/newshoes/masks'} for k,v in items.items()}}
    (ROOT/'data/source/layout'/f'{zid}.json').write_text(json.dumps(layout,indent=1)+'\n')
    Image.fromarray(ov).save(outdir/f'{zid}-masks-overlay.jpg',quality=90)
    ymin=min(b[1] for b in items.values())
    cfg={"surface_items":{"surfaces":[{"bbox":[0,372,1536,1024],"donor_x":[20,90],"donor_y":[800,900]}],
      "floor":{"fit_top":min(250,ymin-20),"blend_top":372,"edge":[0,372,0.0],"span":[200,1400],"exclude":[0,0,2,2],"grain_donor":[20,420,120,520]},
      "generated_plate":f"data/source/generated/shoes-zone-{letter}-rug-v240.webp","generated_collar":2,
      "external_masks":{i:f'data/source/newshoes/masks/{i}.png' for i in items},"external_mask_core":15}}
    lp=ROOT/'data/source/layers-v23.json';raw=lp.read_text();l23=json.loads(raw);l23[zid]=cfg
    lp.write_text(json.dumps(l23,indent=2)+('\n' if raw.endswith('\n') else ''))
    print(json.dumps({zid:items,**info}))

ZNUM={'b':'test image','c':'zone-01','d':'zone-02','e':'zone-03','f':'zone-04','g':'zone-05','h':'zone-06','i':'zone-07','j':'zone-08','k':'zone-09'}
def register(letters):
    """Registers zones in zones.json / generated-v22.json (painting, plate, card cutouts)."""
    items=json.loads((SRC/'items.json').read_text())['items']
    zp=ROOT/'data/source/zones.json';zraw=zp.read_text();zones=json.loads(zraw)
    gp=ROOT/'data/source/generated-v22.json';graw=gp.read_text();gen=json.loads(graw)
    for L in letters:
        zid=f'shoes-{L}';ids=[i['id'] for i in items if i['zone']==zid]
        zones.setdefault('shoes',{})[L]=ids
        gen['zones'][zid]=f'data/source/newshoes/zones/{zid}.webp'
        for i in ids: gen['cutouts'][i]=f'data/source/newshoes/cutouts/{i}.webp'
        gen['plates'][zid]={"file":f"data/source/generated/shoes-zone-{L}-rug-v240.webp","version":"v2.4.0",
          "use":f"empty-rug plate for {ZNUM[L]} ({zid}); registered/colour-matched and used only in a collar around the pairs and their shadows",
          "generator":f"Codex CLI imagegen (ChatGPT plan), refs: {ZNUM[L]} painting + two rug crops","metadata":"stripped",
          "raw":f"work/newshoes/rollout/gen/shoes-zone-{L}-rug-v240.png (not in git)","approved":"2026-11-17 by user via PM"}
    zones['shoes']=dict(sorted(zones['shoes'].items()))
    zp.write_text(json.dumps(zones,indent=1)+('\n' if zraw.endswith('\n') else ''))
    gp.write_text(json.dumps(gen,indent=2)+('\n' if graw.endswith('\n') else ''))

if __name__=='__main__':
    cmd=sys.argv[1] if len(sys.argv)>1 else 'cutouts'
    if cmd=='cutouts':run_cutouts(sys.argv[2].split(',') if len(sys.argv)>2 else None,sys.argv[3] if len(sys.argv)>3 else None)
    elif cmd=='auto':run_zone_auto(sys.argv[2],sys.argv[3],sys.argv[4] if len(sys.argv)>4 else None)
    elif cmd=='register':register(sys.argv[2].split(','))
    elif cmd=='zone':run_zone(plate_png=sys.argv[2] if len(sys.argv)>2 else None)
