"""Surface-standing items: deterministic cutouts and same-surface donor repair.

Opt-in geometry lives in layers-v23.json; no changes to wall/rack extraction.
The pair is the selectable item, so overlap within each pair stays intact.
"""
import cv2
import numpy as np

class SurfaceMasks(dict):
    pass


def generated_plate(rgb, union, config):
    """v2.3.1: the empty surface comes from a generated edit of this zone (data/source/generated).
    It is registered to the painting, colour-matched at low frequency, and used only inside a
    feathered collar around the items; every other pixel stays the original painting."""
    from pathlib import Path
    from PIL import Image
    h,w=rgb.shape[:2]
    src=Path(__file__).resolve().parents[1]/config['generated_plate']
    gen=np.array(Image.open(src).convert('RGB').resize((w,h),Image.Resampling.LANCZOS))
    known=(cv2.dilate(union,np.ones((41,41),np.uint8))==0)
    near=cv2.dilate(union,np.ones((141,141),np.uint8))>0
    known[:config['floor']['fit_top']]=False
    g0=cv2.GaussianBlur(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32),(0,0),1.2)
    g1=cv2.GaussianBlur(cv2.cvtColor(gen,cv2.COLOR_RGB2GRAY).astype(np.float32),(0,0),1.2)
    warp=np.eye(2,3,dtype=np.float32)
    try:
        _,warp=cv2.findTransformECC(g0,g1,warp,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,200,1e-6),np.uint8(known)*255,5)
    except cv2.error:
        pass
    gen=cv2.warpAffine(gen,warp,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT)
    # Low-frequency colour/illumination match from clean (non-item, non-shadow) pixels.
    ratio=(rgb.astype(float)+1)/(gen.astype(float)+1)
    clean=known&(np.abs(np.log(ratio)).max(2)<.25)
    # first pass far-field gain to spot darker (shadowed) pixels near the items and exclude them
    f=cv2.GaussianBlur(clean.astype(float),(0,0),sig0:=config.get('generated_match_sigma',18)*4)
    gf=np.exp(cv2.GaussianBlur(np.log(ratio)*clean[:,:,None],(0,0),sig0)/np.maximum(f[:,:,None],1e-3))
    lr0=cv2.GaussianBlur(rgb.astype(float),(0,0),4)/np.maximum(cv2.GaussianBlur(gen*gf,(0,0),4),1)
    clean&=~(near&(lr0.max(2)<.95))
    sig=config.get('generated_match_sigma',18)
    wgt=cv2.GaussianBlur(clean.astype(float),(0,0),sig)
    num=cv2.GaussianBlur(np.log(ratio)*clean[:,:,None],(0,0),sig)
    gain=np.exp(num/np.maximum(wgt[:,:,None],1e-3))
    far=cv2.GaussianBlur(clean.astype(float),(0,0),sig*4)
    gnum=cv2.GaussianBlur(np.log(ratio)*clean[:,:,None],(0,0),sig*4)
    gain_far=np.exp(gnum/np.maximum(far[:,:,None],1e-3))
    t=np.clip(wgt/.25,0,1)[:,:,None]
    gain=gain*t+gain_far*(1-t)
    gen=np.clip(gen*gain,0,255)
    # Cast shadows of the removed items are also replaced: smooth, colour-neutral darkening of the
    # painting relative to the matched empty surface, connected to the items.
    lr=cv2.GaussianBlur(rgb.astype(float),(0,0),4)/np.maximum(cv2.GaussianBlur(gen,(0,0),4),1)
    dark=(lr.max(2)<config.get('generated_shadow_ratio',.94))&(np.ptp(lr,axis=2)<.12)
    dark&=cv2.dilate(union,np.ones((121,121),np.uint8))>0
    dark[:config['floor']['fit_top']]=False
    dark=cv2.morphologyEx(np.uint8(dark)*255,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    n,ll,st,_=cv2.connectedComponentsWithStats(dark)
    touch=cv2.dilate(union,np.ones((15,15),np.uint8))>0
    shadow=np.isin(ll,[k for k in range(1,n) if np.any(touch&(ll==k))])
    collar=config.get('generated_collar',9)
    region=cv2.dilate(np.uint8((union>0)|shadow)*255,np.ones((2*collar+1,2*collar+1),np.uint8)).astype(float)
    alpha=cv2.GaussianBlur(region,(0,0),collar/3)/255
    alpha=np.maximum(alpha,(union>0).astype(float))
    alpha[:config['floor']['fit_top']-20]=0
    return np.uint8(np.clip(rgb*(1-alpha[:,:,None])+gen*alpha[:,:,None],0,255))


def reconstruct(rgb, boxes, config):
    h,w=rgb.shape[:2]
    yy,xx=np.indices((h,w))
    floor_config=config['floor']
    floor_start=floor_config['fit_top']
    floor_split=floor_config['blend_top']
    masks=SurfaceMasks(); masks.metrics={}; reasons={}
    for iid,(x,y,r,b) in boxes.items():
        if iid in config.get('external_masks',{}):
            # V2.4.0: the pair silhouette comes from a prepared mask (tools/newshoes_prep.py: painting vs the
            # empty-rug plate). Only zones that list external_masks take this path; shoes-a is unchanged.
            from pathlib import Path
            from PIL import Image
            ext=np.array(Image.open(Path(__file__).resolve().parents[1]/config['external_masks'][iid]).convert('L'))
            masks[iid]=np.uint8(ext>127)*255;reasons[iid]=[]
            continue
        pad=16
        x0,y0,x1,y1=max(0,x-pad),max(0,y-pad),min(w,r+pad),min(h,b+pad)
        crop=rgb[y0:y1,x0:x1]
        gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
        # Leather/suede is darker than the rug. GrabCut retains cream insoles
        # enclosed by the dark uppers, while rejecting the rug between shoes.
        labels=np.full(gray.shape,cv2.GC_PR_BGD,np.uint8)
        labels[gray<config.get('foreground_dark',105)]=cv2.GC_PR_FGD
        labels[gray<65]=cv2.GC_FGD
        labels[:y-y0]=0; labels[b-y0:]=0; labels[:,:x-x0]=0; labels[:,r-x0:]=0
        cv2.setRNGSeed(0)
        cv2.grabCut(crop,labels,None,np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_MASK)
        m=np.uint8((labels==1)|(labels==3))*255
        n,ll,stats,_=cv2.connectedComponentsWithStats(m)
        m=np.uint8(np.isin(ll,[k for k in range(1,n) if stats[k,4]>max(50,stats[1:,4].max()*.02)]))*255
        # Closed light insoles and tiny lettering holes belong to the shoe.
        # Large openings within lace loops remain transparent.
        n,holes,areas,_=cv2.connectedComponentsWithStats(255-m)
        for k in range(1,n):
            hx,hy,hw,hh,area=areas[k]
            if hx==0 or hy==0 or hx+hw==m.shape[1] or hy+hh==m.shape[0]:continue
            if area<250 or area>1000:m[holes==k]=255
        # Detached buckle centres are small dark components. Recover them
        # only when enclosed by gold near the main shoe, never rug flecks.
        red,green,blue=crop.astype(int).transpose(2,0,1)
        gold=(red-blue>45)&(red-green>12)&(green-blue>18)&(red>130)
        nearby=cv2.dilate(m,np.ones((51,51),np.uint8))>0
        gold_near=cv2.dilate(np.uint8(gold),np.ones((9,9),np.uint8))>0
        for k in range(1,len(stats)):
            component=ll==k
            if stats[k,4]>=100 and np.any(component&nearby&gold_near):m[component]=255
        attached=cv2.dilate(m,np.ones((9,9),np.uint8))>0
        additions=np.uint8(gold&attached)*255
        n,gl,gs,_=cv2.connectedComponentsWithStats(additions)
        for k in range(1,n):
            if gs[k,4]>=20:m[gl==k]=255
        full=np.zeros((h,w),np.uint8);full[y0:y1,x0:x1]=m
        masks[iid]=full;reasons[iid]=[]
    union=np.maximum.reduce(list(masks.values()))
    # Donors follow the rug's perspective. Its top/bottom borders use the
    # same source row; the field uses an interior patch without edge motifs.
    background=rgb.copy()
    for surface in config['surfaces']:
        x,y,r,b=surface['bbox']; a,z=surface['donor_x']
        donor_y=surface.get('donor_y',[y,b])
        strip=rgb[donor_y[0]:donor_y[1],a:z].copy()
        tile=np.concatenate((strip,strip[:,::-1]),axis=1)
        tile=np.concatenate((tile,tile[::-1]),axis=0)
        sx=(np.arange(x,r)-a)%tile.shape[1]
        sy=(np.arange(y,b)[:,None]-donor_y[0]-np.round((np.arange(x,r)[None,:]-(a+z)/2)*surface.get('slope',0))).astype(int)%tile.shape[0]
        background[y:b,x:r]=tile[sy,sx[None,:]]
    # Rebuild only missing texture. For exposed cast shadows retain the real
    # rug pattern and remove its low-frequency darkening with a smooth gain.
    # Normalised convolution excludes leather from the illumination fit.
    # Blurring the unmasked painting would mistake the dark shoes for a
    # shadow and introduce a bright halo when that fit was inverted.
    valid=(union==0).astype(float)
    clean=cv2.GaussianBlur(valid,(0,0),7)
    source_light=cv2.GaussianBlur(rgb.astype(float)*valid[:,:,None],(0,0),7)/np.maximum(clean[:,:,None],.001)
    donor_light=cv2.GaussianBlur(background.astype(float),(0,0),25)
    near=cv2.dilate(union,np.ones((181,181),np.uint8))>0
    gain=np.clip(donor_light/np.maximum(source_light,1),1,1.8)
    weight=cv2.GaussianBlur(near.astype(float),(0,0),12)
    weight[:floor_split]=0
    gain=1+(gain-1)*weight[:,:,None]
    plate=np.uint8(np.clip(rgb.astype(float)*gain,0,255))
    plate[union>0]=background[union>0]
    repair=cv2.dilate(union,np.ones((11,11),np.uint8))
    repair[:floor_split]=0
    repair[-1]=0;repair[:,0]=0;repair[:,-1]=0
    ys,xs=np.where(repair>0)
    center=((int(xs.min())+int(xs.max()))//2,(int(ys.min())+int(ys.max()))//2)
    plate=cv2.seamlessClone(background,plate,repair,center,cv2.NORMAL_CLONE)
    # Fit the complete exposed floor band. Rebuilding only the collar-shaped
    # holes leaves a visible smooth patch in a textured floor.
    origin_x,origin_y,slope=floor_config['edge']
    floor_edge=origin_y+(xx-origin_x)*slope
    band=(yy>=floor_start)&(yy<floor_edge)&(xx>floor_config['span'][0])&(xx<floor_config['span'][1])
    fx,fy,fr,fb=floor_config['exclude']
    furniture=(xx>=fx)&(xx<fr)&(yy>=fy)&(yy<fb)
    known=band&~furniture&~(cv2.dilate(union,np.ones((61,61),np.uint8))>0)
    features=[np.ones((h,w)),xx/w,yy/h,(xx/w)**2,(xx/w)*(yy/h)]
    A=np.stack([f[known] for f in features],axis=1)
    coeff=np.linalg.lstsq(A,rgb[known].astype(float),rcond=None)[0]
    floor_model=sum(f[:,:,None]*c for f,c in zip(features,coeff))
    # Continue genuine floor grain independently of its broad illumination.
    gx,gy,gr,gb=floor_config['grain_donor']
    donor=rgb[gy:gb,gx:gr].astype(float)
    grain=donor-cv2.GaussianBlur(donor,(0,0),8)
    floor_model+=grain[(yy-gy)%(gb-gy),(xx-gx)%(gr-gx)]
    floor_model=np.uint8(np.clip(floor_model,0,255))
    floor_dest=rgb.copy()
    floor_dest[union>0]=floor_model[union>0]
    floor_repair=cv2.dilate(union,np.ones((21,21),np.uint8))
    floor_repair[-1]=0;floor_repair[:,0]=0;floor_repair[:,-1]=0
    ys,xs=np.where(floor_repair>0)
    center=((int(xs.min())+int(xs.max()))//2,(int(ys.min())+int(ys.max()))//2)
    floor_result=cv2.seamlessClone(floor_model,floor_dest,floor_repair,center,cv2.NORMAL_CLONE)
    floor_alpha=cv2.GaussianBlur(cv2.dilate(union,np.ones((9,9),np.uint8)),(9,9),2).astype(float)/255
    floor_alpha[union>0]=1
    floor_alpha[yy>=floor_edge]=0
    plate=np.uint8(plate*(1-floor_alpha[:,:,None])+floor_result*floor_alpha[:,:,None])
    # Match broad illumination after texture blending, so a cast shadow at
    # the blend boundary cannot propagate a shoe-shaped stain into the fill.
    average=cv2.GaussianBlur(plate.astype(float),(0,0),25)
    target=cv2.GaussianBlur(background.astype(float),(0,0),40)
    correction=np.clip(target/np.maximum(average,1),.8,1.5)
    correction[:floor_split]=1
    plate=np.uint8(np.clip(plate*(1+(correction-1)*weight[:,:,None]),0,255))
    if config.get('generated_plate'):
        plate=generated_plate(rgb,union,config)
        masks.generated=True
    comp=plate.astype(float)
    for iid,m in masks.items():
        a=cv2.GaussianBlur(m,(3,3),.55).astype(float)/255
        comp=comp*(1-a[:,:,None])+rgb*a[:,:,None]
    ratio=(rgb.astype(float)+1)/(plate.astype(float)+1)
    shadow=(ratio.max(2)<.985)&(ratio.min(2)>.35)&(np.ptp(ratio,axis=2)<.10)
    lost=(np.linalg.norm(comp-rgb,axis=2)>30)&~shadow
    for iid,(x,y,r,b) in boxes.items():
        m=masks[iid]>0
        rows,cols=np.where(m)
        if not len(rows) or min(rows)>y+8 or max(rows)<b-12 or min(cols)>x+12 or max(cols)<r-12:
            reasons[iid].append('silhouette check: incomplete surface item')
        error=float(lost[max(0,y-30):min(h,b+30),max(0,x-30):min(w,r+30)].mean())
        masks.metrics[iid]={'rest_err':round(error,4)}
        if error>.01:
            reasons[iid].append('rest check: lost item or surface mismatch')
        core=m
        if iid in config.get('external_masks',{}):
            # prepared masks carry a deliberate safety margin of plain rug; judge removal on the object core only
            k=config.get('external_mask_core',15)
            core=cv2.erode(np.uint8(m)*255,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))>0
        unchanged=(np.linalg.norm(rgb.astype(float)-plate.astype(float),axis=2)<14)&core
        if unchanged.sum()/max(1,core.sum())>.03:
            reasons[iid].append('removal check: unchanged foreground')
    # Never export an empty hole where a rejected layer will be omitted.
    if any(reasons.values()):
        masks.preview_plate=plate.copy()
        for iid in reasons:
            if not reasons[iid]:reasons[iid]=['surface integrity check: neighbouring item rejected']
        plate=rgb.copy()
    return plate,masks,reasons
