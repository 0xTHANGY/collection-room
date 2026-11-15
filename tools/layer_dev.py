#!/usr/bin/env python3
"""Fast per-zone layer/plate check (v2.3). Usage: layer_dev.py ZID [ZID...] | --cat CAT | --all
Runs the same zone loading and reconstruct() as build_v2.py for the given zones only, writes review
images + metrics to work/v2.3/review/, and prints a pass table. Writes nothing under assets/ or data/.
Metrics per passing item:
  rest_err   (shadows excluded) fraction of mask-bbox pixels where plate+layers (at rest) differs from the original painting (>30 RGB)
             -> catches layers cropped too tightly (lost pixels get repainted as wall/rail on the plate)
  wall_in_layer  fraction of layer pixels that look like plain wall (match the plate within 14 RGB) and touch the layer edge
             -> catches convex-hull wedges / background dragged into the layer
"""
import csv, json, sys, time
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1]; FINAL=ROOT.parent/'final'
sys.path.insert(0,str(ROOT/'tools')); sys.path.insert(0,str(ROOT/'data/source'))
import v21_images
cv2.setNumThreads(1)
OUT=ROOT/'work/v2.3/review'; OUT.mkdir(parents=True,exist_ok=True)
def load(zid):
    cat,letter=zid.rsplit('-',1)
    GEN=json.loads((ROOT/'data/source/generated-v22.json').read_text())
    finals={(r['category'],r['zone']):r for r in csv.DictReader((FINAL/'zone-finals.csv').open())}
    src=ROOT/GEN['zones'][zid] if zid in GEN['zones'] else FINAL/finals[cat,letter]['path_in_zip']
    im=Image.open(src).convert('RGB'); ow,oh=im.size; s=min(1,1600/ow)
    im=im.resize((round(ow*s),round(oh*s)),Image.Resampling.LANCZOS); rgb=np.array(im); h,w=rgb.shape[:2]
    ids=json.loads((ROOT/'data/source/zones.json').read_text())[cat][letter]
    L=json.loads((ROOT/'data/source/layout'/f'{zid}.json').read_text()); lw,lh=L['image_size']
    boxes={}
    for i in ids:
        e=L['items'].get(i)
        if e: x0,y0,x1,y1=e['bbox']; boxes[i]=[round(x0*w/lw),round(y0*h/lh),round(x1*w/lw),round(y1*h/lh)]
    if zid=='shirts-a':
        from overrides_v21 import neutral_shirts
        rgb=neutral_shirts(rgb,{})
    return cat,rgb,boxes
def check(zid):
    t=time.time(); cat,rgb,boxes=load(zid)
    fn=getattr(v21_images,'reconstruct_zone',None)
    if fn: base,masks,reasons=fn(rgb,boxes,zid)
    else: base,masks,reasons=v21_images.reconstruct(rgb,boxes,zid)
    h,w=rgb.shape[:2]; comp=base.astype(float).copy()
    alphas={}
    for i,m in masks.items():
        if reasons[i]: continue
        a=cv2.GaussianBlur(m,(3,3),.55).astype(float)/255; alphas[i]=a
    order=getattr(v21_images,'LAYER_ORDER',None)
    for i,a in alphas.items(): comp=comp*(1-a[:,:,None])+rgb*a[:,:,None]
    diff=np.linalg.norm(comp-rgb,axis=2)>30
    # Contact/drop shadows are intentionally dropped (the site renders its own drop-shadow): a pixel that is
    # a uniform darkening of the plate is not counted as lost content.
    ratio=(rgb.astype(float)+1)/(base.astype(float)+1)
    shadow=(ratio.max(2)<.985)&(ratio.min(2)>.35)&(ratio.max(2)-ratio.min(2)<.10)
    diff&=~shadow
    res={}
    for i,(x,y,r,b) in boxes.items():
        d=dict(passed=not reasons[i],reasons=reasons[i])
        if not reasons[i]:
            px,py=int((r-x)*.15),int((b-y)*.15)
            reg=np.zeros((h,w),bool);reg[max(0,y-py):b+py,max(0,x-px):r+px]=True
            d['rest_err']=round(float(diff[reg].mean()),4)
            m=masks[i]>0; wallish=(np.linalg.norm(rgb.astype(float)-base.astype(float),axis=2)<14)&m
            n,lab=cv2.connectedComponents(np.uint8(wallish))
            edge=m&~cv2.erode(np.uint8(m),np.ones((3,3),np.uint8)).astype(bool)
            touching=set(np.unique(lab[edge&wallish]))-{0}
            d['wall_in_layer']=round(float(np.isin(lab,list(touching)).sum()/max(1,m.sum())),4) if touching else 0.0
            d['suspect']=d['rest_err']>.01 or d['wall_in_layer']>.03
        res[i]=d
    # review sheets
    Image.fromarray(base).save(OUT/f'{zid}-plate.jpg',quality=88)
    Image.fromarray(np.uint8(np.clip(comp,0,255))).save(OUT/f'{zid}-rest.jpg',quality=88)
    dm=rgb.copy(); dm[diff]=[255,0,0]; Image.fromarray(dm).save(OUT/f'{zid}-resterr.jpg',quality=85)
    tiles=[]
    for i,m in masks.items():
        yy,xx=np.where(m>0)
        if not len(xx): continue
        crop=Image.fromarray(np.dstack([rgb,m])).crop((xx.min(),yy.min(),xx.max()+1,yy.max()+1)); crop.thumbnail((230,230))
        t2=Image.new('RGB',(240,260),(255,0,255) ); t2.paste(crop,((240-crop.width)//2,0),crop)
        ImageDraw.Draw(t2).text((3,234),f"{i} {'PASS' if not reasons[i] else 'FAIL'} {'SUSPECT' if res[i].get('suspect') else ''}",fill='black')
        ImageDraw.Draw(t2).text((3,246),'; '.join(reasons[i])[:38],fill='black'); tiles.append(t2)
    if tiles:
        cols=5; sh=Image.new('RGB',(cols*240,260*((len(tiles)+cols-1)//cols)),'white')
        for k,t2 in enumerate(tiles): sh.paste(t2,(k%cols*240,k//cols*260))
        sh.save(OUT/f'{zid}-layers.jpg',quality=88)
    (OUT/f'{zid}-check.json').write_text(json.dumps(res,indent=1))
    p=sum(v['passed'] for v in res.values()); s=[i for i,v in res.items() if v.get('suspect')]
    print(f'{zid}: {p}/{len(res)} pass  suspect={s}  ({time.time()-t:.0f}s)',flush=True)
    for i,v in res.items():
        if v.get('suspect'): print('   ',i,{k:v[k] for k in ('rest_err','wall_in_layer')})
    return res
if __name__=='__main__':
    a=sys.argv[1:]; zs=json.loads((ROOT/'data/source/zones.json').read_text())
    if a[:1]==['--all']: a=[f'{c}-{l}' for c in zs for l in sorted(zs[c])]
    elif a[:1]==['--cat']: a=[f'{a[1]}-{l}' for l in sorted(zs[a[1]])]
    for z in a: check(z)
