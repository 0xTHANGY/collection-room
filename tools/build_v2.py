#!/usr/bin/env python3
"""Deterministic, rerunnable v2 asset/catalog builder. Writes only inside site/.
Requires Pillow, numpy and OpenCV; no browser build step. Source layouts are optional.
Reproducible from sources only: it never reads its own previous outputs (assets/zones, assets/items/v2,
data/catalog.json) and never skips work based on file mtimes. Outputs that cannot be recomputed are
listed in data/source/frozen-assets.json and copied through byte-for-byte from data/source/frozen/.
"""
import csv, json, os, re, shutil, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"data/source"))
from overrides_v21 import neutral_shirts
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from v21_images import reconstruct, review, fix_card, REVIEW
ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT.parent / 'final'
LABELS = [('jackets','外套','Jackets'),('shirts','衬衫','Shirts'),('tshirts','T恤','T-shirts'),('bottoms','下装','Bottoms'),('dresses','连衣裙','Dresses'),('sweaters','针织衫','Sweaters'),('hoodies','卫衣','Hoodies'),('bags','包袋','Bags'),('totes','托特包','Totes'),('accessories','配饰','Accessories'),('shoes','鞋履','Shoes'),('hats','帽子','Hats')]
written=set()
FROZEN_LIST=ROOT/'data/source/frozen-assets.json'
FROZEN=json.loads(FROZEN_LIST.read_text())['assets'] if FROZEN_LIST.exists() else {}
def save(im, path, quality=80):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
    if path in FROZEN:
        # Explicit frozen source: copy the recorded bytes, never re-encode.
        shutil.copyfile(ROOT/FROZEN[path],p);written.add(path);return
    # A new pixel-only image prevents metadata propagation.
    clean=Image.fromarray(np.asarray(im))
    temporary=p.with_suffix('.webp.tmp')
    clean.save(temporary,'WEBP',quality=quality,method=4)
    temporary.replace(p)
    written.add(path)
def variants(im, stem, sizes, longest=False):
    result=[]
    for size in sizes:
        scale=min(1,size/(max(im.size) if longest else im.width))
        out=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.LANCZOS)
        path=f'{stem}-{size}.webp'
        save(out,path)
        result.append({'src':path,'width':out.width,'height':out.height})
    return result
def rect(box,w,h):
    x,y,r,b=box
    return dict(left=x/w*100,top=y/h*100,width=(r-x)/w*100,height=(b-y)/h*100)
def intersect(a,b):
    return min(a[2],b[2])-max(a[0],b[0])>1e-6 and min(a[3],b[3])-max(a[1],b[1])>1e-6

def stamp_version():
    """Fill the footer version in index.html from data/source/site.json (the one place to change it)."""
    version=json.loads((ROOT/'data/source/site.json').read_text())['version']
    page=ROOT/'index.html';html=page.read_text()
    new,n=re.subn(r'(<span lang="en" id="site-version">)[^<]*(</span>)',lambda m:m.group(1)+'COLLECTION ROOM · '+version+m.group(2),html)
    if n!=1: raise SystemExit('footer version span #site-version not found exactly once in index.html')
    if new!=html: page.write_text(new)
    print('footer version:',version)
def main():
    stamp_version()
    cv2.setNumThreads(1);cv2.setRNGSeed(42)
    mapping=json.loads((ROOT/'data/source/zones.json').read_text())
    manifest=list(csv.DictReader((FINAL/'manifest.csv').open()))
    finals={(r['category'],r['zone']):r for r in csv.DictReader((FINAL/'zone-finals.csv').open())}
    hub={i['id']:i['display_name'] for i in json.loads((ROOT/'data/items.json').read_text())}
    fixes=json.loads((ROOT/'data/source/cutout-fixes.json').read_text())
    gate={}
    for old in (ROOT/'assets/items/v2').glob('*-800.webp'): old.unlink()
    GEN=json.loads((ROOT/'data/source/generated-v22.json').read_text()) if (ROOT/'data/source/generated-v22.json').exists() else {'zones':{},'cutouts':{}}
    catalog={'version':'2.3','categories':[],'zones':[],'items':[]}
    missing=[];fallback=[];counts={}
    for order,(cat,zh,en) in enumerate(LABELS):
        source=json.loads((FINAL/cat/'docs'/f'{cat}-items.json').read_text())
        lookup={i['id']:i for i in source['items']}
        extra=ROOT/'data/source/newshoes/items.json'   # V2.4.0 second source (new shoes); same fields as final/<cat>/docs
        if extra.exists(): lookup.update({i['id']:i for i in json.loads(extra.read_text())['items'] if i['zone'].startswith(cat+'-')})
        letters=sorted(mapping[cat])
        catalog['categories'].append(dict(id=cat,label=zh,label_en=en,order=order,cover_zone=f'{cat}-{letters[0]}',count=sum(len(v) for v in mapping[cat].values())))
        for letter in letters:
            zid=f'{cat}-{letter}';ids=mapping[cat][letter]
            zone_src=ROOT/GEN['zones'][zid] if zid in GEN['zones'] else FINAL/finals[cat,letter]['path_in_zip']
            painting=Image.open(zone_src).convert('RGB')
            ow,oh=painting.size;scale=min(1,1600/ow)
            painting=painting.resize((round(ow*scale),round(oh*scale)),Image.Resampling.LANCZOS)
            rgb=np.array(painting);h,w=rgb.shape[:2]
            lp=ROOT/'data/source/layout'/f'{zid}.json'
            layout=json.loads(lp.read_text()) if lp.exists() else None
            if not layout: missing.append(zid);print('WARNING missing layout:',zid)
            boxes={};layers={};masks={};bad=set()
            if layout:
                lw,lh=layout['image_size']
                for iid in ids:
                    entry=layout['items'].get(iid)
                    if not entry: continue
                    x0,y0,x1,y1=entry['bbox'];box=[round(x0*w/lw),round(y0*h/lh),round(x1*w/lw),round(y1*h/lh)]
                    x0,y0,x1,y1=box;boxes[iid]=box
                    pad=max(12,round(max(x1-x0,y1-y0)*.12))
                    rx0=max(0,x0-pad);ry0=max(0,y0-pad);rx1=min(w,x1+pad);ry1=min(h,y1+pad)
                    roi=rgb[ry0:ry1,rx0:rx1]
                    mask=np.zeros((h,w),np.uint8)
                    if cat in ('hats','bags','accessories'):
                        mask[y0:y1,x0:x1]=255  # replaced by wall-model masks below
                    else:
                        # Conservative fallback mask, always computed from the source painting.
                        local=np.zeros(roi.shape[:2],np.uint8)
                        cv2.grabCut(cv2.cvtColor(roi,cv2.COLOR_RGB2BGR),local,(x0-rx0,y0-ry0,x1-x0,y1-y0),np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_RECT)
                        mask[ry0:ry1,rx0:rx1]=np.uint8((local==1)|(local==3))*255
                    # Reject fragmented or implausibly sparse/solid extractions.
                    n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
                    if n>1:
                        largest=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);mask=np.uint8(labels==largest)*255
                    coverage=np.count_nonzero(mask)/((x1-x0)*(y1-y0))
                    if coverage<.18 or coverage>.95: bad.add(iid)
                    masks[iid]=mask
                if zid=='shirts-a':
                    # Reproducible: protect garments with a frozen mask (data/source/shirts-a-neutral-protect.png,
                    # snapshot of the layer masks at bcd8174), never with this build's own previous layer outputs.
                    protect=np.array(Image.open(ROOT/'data/source/shirts-a-neutral-protect.png').convert('L'))
                    assert protect.shape==rgb.shape[:2],protect.shape
                    rgb=neutral_shirts(rgb,{'frozen':np.uint8(protect>127)*255})
                    painting=Image.fromarray(rgb)
                # v2.3: every category goes through the plate reconstruction + per-item gate.
                prior_masks=masks
                reconstructed,masks,reasons=reconstruct(rgb,boxes,zid)
                ENABLED=json.loads((ROOT/'data/source/layers-v23.json').read_text())['enabled_categories']
                if cat not in ENABLED:
                    # Category not yet reviewed in v2.3: keep the v2.2 behaviour (painting + fallback layers).
                    reasons={iid:['wall reconstruction not validated for this background'] for iid in boxes}
                    reconstructed=rgb
                bad={iid for iid in boxes if reasons[iid]}
                base=Image.fromarray(reconstructed)
                if cat not in ('hats','bags','accessories'):
                    # Clothing/shoes items that still fail keep their earlier conservative layer.
                    for iid in bad: masks[iid]=prior_masks[iid]
                gate[zid]={iid:dict(passed=iid not in bad,reasons=reasons[iid]) for iid in boxes}
                review(zid,np.array(base),rgb,masks,reasons)
                for iid,mask in masks.items():
                    # Include the inpaint boundary under opaque source pixels, feather outward.
                    alpha=cv2.GaussianBlur(mask,(3,3),.55)
                    yy,xx=np.where(alpha>0)
                    if not len(xx): bad.add(iid);continue
                    x0,x1=xx.min(),xx.max()+1;y0,y1=yy.min(),yy.max()+1
                    path=f'assets/zones/layers/{iid}.webp'
                    layer_rgb=getattr(masks,'images',{}).get(iid,rgb)
                    save(Image.fromarray(np.dstack([layer_rgb,alpha])[y0:y1,x0:x1]),path,82)
                    motion='hang' if cat=='bottoms' else (layout['items'][iid].get('motion') or ('lift' if cat in ('shoes','accessories') or boxes[iid][1]/h>.7 else 'hang'))  # v2.3: bottoms hang from hangers -> same swing as shirts/jackets (user decision)
                    lr=rect([x0,y0,x1,y1],w,h)
                    layers[iid]=dict(image=path,**lr,rect=lr,pivot={'x':50,'y':0 if motion=='hang' else 50},pivotX=50,pivotY=0 if motion=='hang' else 50,motion=motion,fallback=iid in bad)
                for depth,iid in enumerate(getattr(masks,'order',[])):
                    if iid in layers: layers[iid]['zIndex']=depth+1
            else: base=painting
            fallback.extend(sorted(bad))
            paths=variants(base,f'assets/zones/{zid}',[1600,1024,640])
            cover=variants(painting,f'assets/zones/{zid}-cover',[640])[0]
            catalog['zones'].append(dict(id=zid,category=cat,letter=letter,images=paths,cover=cover,natural_size=[ow,oh],aspect=ow/oh,has_layout=bool(layout),items=ids))
            # Inset then split overlapping rectangles along centre-to-centre axis.
            spots={iid:[b[0]+(b[2]-b[0])*.025,b[1]+(b[3]-b[1])*.025,b[2]-(b[2]-b[0])*.025,b[3]-(b[3]-b[1])*.025] for iid,b in boxes.items()}
            for _ in range(len(spots)**2+1):
                changed=False
                for i,a in enumerate(spots):
                    for b in list(spots)[i+1:]:
                        A,B=spots[a],spots[b]
                        if not intersect(A,B):continue
                        ca=[(A[0]+A[2])/2,(A[1]+A[3])/2];cb=[(B[0]+B[2])/2,(B[1]+B[3])/2]
                        axis=0 if abs(ca[0]-cb[0])>=abs(ca[1]-cb[1]) else 1
                        low,high=(A,B) if ca[axis]<=cb[axis] else (B,A)
                        mid=(max(low[axis],high[axis])+min(low[axis+2],high[axis+2]))/2
                        low[axis+2]=mid-.25;high[axis]=mid+.25;changed=True
                if not changed:break
            pairs=sum(intersect(a,b) for i,a in enumerate(spots.values()) for b in list(spots.values())[i+1:]);counts[zid]=pairs
            assert pairs==0,zid
            print(zid,'overlap pairs:',pairs)
            for iid in ids:
                src=lookup[iid];cut=FINAL/cat/'cutouts'/f'{iid}.png'
                if not cut.exists() and iid not in GEN['cutouts']:cut=FINAL/next(r['path_in_zip'] for r in manifest if r['category']==cat and r['item_id']==iid.split('-')[-1])
                if iid in GEN['cutouts']:cut=ROOT/GEN['cutouts'][iid]
                original=Image.open(cut).convert('RGBA')
                fixed=fix_card(original,iid,fixes,rgb,boxes.get(iid))
                cards=variants(fixed,f'assets/items/v2/{iid}',[600,400],True)
                brand=src.get('brand') or ''
                if src.get('brand_uncertain') or re.search('不确定|unknown',brand,re.I):brand='Unknown'
                catalog['items'].append(dict(id=iid,category=cat,zone=zid,number=src.get('number') or iid.split('-')[-1],name=hub.get(iid,src['name']),display_name=hub.get(iid,src['name']),brand=brand,type=src.get('type',''),story='故事待补充',image=cards[0]['src'],card_images=cards,hotspot=rect(spots[iid],w,h) if iid in spots else None,layer=layers.get(iid)))
    (REVIEW/'gate.json').write_text(json.dumps(gate,indent=2))
    (REVIEW/'gate-table.md').write_text('Zone | Passed/total | Failures\n---|---|---\n'+'\n'.join(z+' | '+str(sum(v['passed'] for v in g.values()))+'/'+str(len(g))+' | '+'; '.join(i+': '+', '.join(v['reasons']) for i,v in g.items() if not v['passed']) for z,g in gate.items()))
    (ROOT/'data/catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+'\n')
    summary=dict(card_images_bytes=sum(p.stat().st_size for p in (ROOT/'assets/items/v2').glob('*.webp')),categories=len(catalog['categories']),zones=len(catalog['zones']),items=len(catalog['items']),asset_bytes_written=sum((ROOT/p).stat().st_size for p in written),asset_files_written=len(written),assets_total_bytes=sum(p.stat().st_size for p in (ROOT/'assets').rglob('*') if p.is_file()),overlap_pairs=counts,zones_lacking_layout=missing,fallback_ids=fallback,first_load_bytes=None)
    # Measure the actual responsive/lazy hub requests at the requested viewport.
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--no-sandbox'])
            page=browser.new_page(viewport={'width':1440,'height':900},device_scale_factor=1)
            urls=set();base_url=os.environ.get('BUILD_PREVIEW_URL','http://127.0.0.1:8765/')
            page.on('response',lambda r:urls.add(r.url) if r.ok and r.url.startswith(base_url) else None)
            page.goto(base_url,wait_until='networkidle');page.wait_for_selector('#room-tiles a')
            files=sorted({u.removeprefix(base_url).split('?')[0] or 'index.html' for u in urls})
            summary['first_load_files']=files
            summary['first_load_bytes']=sum((ROOT/f).stat().st_size for f in files if (ROOT/f).is_file())
            browser.close()
    except Exception as error:
        print('WARNING first-load measurement unavailable (start preview on port 8765):',error)
    (ROOT/'data/build-v2-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
