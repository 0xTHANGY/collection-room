"""Deterministic v2.1 wall reconstruction and documented card overrides."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
REVIEW=Path('/workspace/collection-room/work/v2.1/review')
REVIEW.mkdir(parents=True,exist_ok=True)

def reconstruct(rgb, boxes, zid, rejected=None):
 rejected = rejected or {}
 h,w=rgb.shape[:2]; outside=np.ones((h,w),bool)
 for x,y,r,b in boxes.values(): outside[max(0,y-5):min(h,b+5),max(0,x-5):min(w,r+5)]=False
 # Neutral wall samples exclude windows, rails, floor and coloured furnishings.
 neutral=(rgb.max(2).astype(float)-rgb.min(2)<22)&(rgb.mean(2)>125)&outside
 yy,xx=np.indices((h,w)); valid=neutral&(xx>w*.12)&(xx<w*.94)&(yy<h*.85)
 model=np.zeros_like(rgb,dtype=float)
 for ch in range(3):
  A=np.column_stack([np.ones(valid.sum()),xx[valid]/w,yy[valid]/h,(xx[valid]/w)**2,(yy[valid]/h)**2])
  vals=rgb[:,:,ch][valid].astype(float)
  for _ in range(3):
   coeff=np.linalg.lstsq(A,vals,rcond=None)[0]; keep=np.abs(A@coeff-vals)<12;A=A[keep];vals=vals[keep]
  model[:,:,ch]=coeff[0]+coeff[1]*xx/w+coeff[2]*yy/h+coeff[3]*(xx/w)**2+coeff[4]*(yy/h)**2
 model=np.clip(model,0,255).astype(np.uint8)
 brown=(rgb[:,:,0].astype(int)>rgb[:,:,2].astype(int)*1.22)&(rgb[:,:,0]>60)&(rgb.mean(2)<155)
 joined=cv2.morphologyEx(brown.astype('uint8'),cv2.MORPH_CLOSE,np.ones((1,max(3,int(w*.025))),np.uint8))
 long_band=cv2.morphologyEx(joined,cv2.MORPH_OPEN,np.ones((1,max(3,int(w*.50))),np.uint8))
 rail=long_band.any(1)
 # Recover complete rail borders from unoccluded wood samples. Long-band
 # detection alone misses lower rail rows interrupted by hat crowns.
 near=cv2.dilate(rail.astype('uint8')[:,None],np.ones((51,1),np.uint8))[:,0]>0
 rail |= near & ((brown&outside).sum(1)>max(20,w*.08))
 rail_core=cv2.dilate(rail.astype('uint8')[:,None],np.ones((9,1),np.uint8))[:,0]>0
 rail=cv2.dilate(rail.astype('uint8')[:,None],np.ones((17,1),np.uint8))[:,0]>0
 masks={};reasons={}
 for iid,(x,y,r,b) in boxes.items():
  delta=np.linalg.norm(rgb.astype(float)-model.astype(float),axis=2)
  m=np.zeros((h,w),np.uint8);m[y:b,x:r]=np.uint8(delta[y:b,x:r]>32)*255;m[rail]=0
  # Exclude neighbour cores before connected-component selection.
  for other,(a,c,d,e) in boxes.items():
   if other==iid:continue
   dx=int((d-a)*.18);dy=int((e-c)*.18);m[c+dy:e-dy,a+dx:d-dx]=0
  # Refine on a padded crop: central foreground seeds distinguish neutral cloth
  # from the adjacent neutral wall, while the wall model supplies probable seeds.
  px=max(0,x-12);py=max(0,y-12);pr=min(w,r+12);pb=min(h,b+12)
  local=np.where(m[py:pb,px:pr]>0,cv2.GC_PR_FGD,cv2.GC_PR_BGD).astype('uint8')
  local[:y-py,:]=0;local[b-py:,:]=0;local[:,:x-px]=0;local[:,r-px:]=0
  cx=(x+r)//2-px;cy=(y+b)//2-py
  local[max(0,cy-4):cy+5,max(0,cx-4):cx+5]=cv2.GC_FGD
  cv2.setRNGSeed(42)
  cv2.grabCut(rgb[py:pb,px:pr],local,None,np.zeros((1,65)),np.zeros((1,65)),2,cv2.GC_INIT_WITH_MASK)
  m[:]=0;m[py:pb,px:pr]=np.uint8((local==1)|(local==3))*255;m[rail]=0
  m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));m[rail]=0
  n,labels,stats,centers=cv2.connectedComponentsWithStats(m)
  reason=[]
  if n>1:
   scores=stats[1:,4]/(1+((centers[1:,0]-(x+r)/2)/(r-x))**2+((centers[1:,1]-(y+b)/2)/(b-y))**2)
   best=1+np.argmax(scores);m=np.uint8(labels==best)*255
  # Hats have no genuine enclosed background gaps.
  if zid.startswith('hats-'):
   contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(m,[cv2.convexHull(c) for c in contours],-1,255,-1)
  # Fill small enclosed pale textile holes, retaining large handle/background gaps.
  inverse=np.uint8(m==0);nn,ll,ss,_=cv2.connectedComponentsWithStats(inverse)
  for k in range(1,nn):
   if ss[k,4] < (r-x)*(b-y)*.08 and ss[k,0]>x and ss[k,1]>y and ss[k,0]+ss[k,2]<r and ss[k,1]+ss[k,3]<b:m[ll==k]=255
  m[rail]=0
  cov=np.count_nonzero(m)/max(1,(r-x)*(b-y))
  if not .15<cov<.98:reason.append('mask coverage out of range')
  foreground_y=np.where(m>0)[0]
  if zid.startswith('bags-') and len(foreground_y) and foreground_y.min()>y+(b-y)*.18:reason.append('missing pale handle or upper silhouette')
  if zid=='hats-h':reason.append('A-frame background requires redrawing')
  if zid in ('hats-b','hats-d'):reason.append('rail geometry reconstruction unvalidated')
  if b>h*.83 or (zid.startswith('bags-') and b>h*.70):reason.append('floor or furnishing reconstruction unsupported')
  # A wall fit cannot remove objects crossing lower wood/furniture surfaces.
  if np.any(rail_core[max(y,int(h*.48)):b+8]):reason.append('bench boundary: wood reconstruction unvalidated')
  reason.extend(rejected.get(iid, []))
  if y<h*.05:reason.append('item touches image boundary')
  masks[iid]=m;reasons[iid]=reason
 # Actual mask collisions, never bbox intersection alone.
 keys=list(masks)
 for i,a in enumerate(keys):
  for b in keys[i+1:]:
   if np.count_nonzero((masks[a]>0)&(masks[b]>0))>12:
    reasons[a].append('touches neighbour mask');reasons[b].append('touches neighbour mask')
 base=rgb.copy()
 removed=np.zeros((h,w),bool)
 for iid,m in masks.items():
  if reasons[iid]:continue
  x,y,r,b=boxes[iid]
  # Remove painted contact shadows as well as the foreground silhouette.
  hole=np.zeros((h,w),np.uint8)
  # Distance feather has a fully reconstructed core, with a broad, smooth
  # transition rather than a Gaussian-blurred rectangular patch.
  hole[max(0,y-24):min(h,b+25),max(0,x-12):min(w,r+12)]=255
  removed |= hole>0
  distance=cv2.distanceTransform(255-hole,cv2.DIST_L2,5)
  blend=np.clip(1-distance/85,0,1)
  blend=blend*blend*(3-2*blend)
  for other in keys:
   if other==iid or not reasons[other]:continue
   # Smooth protection follows the failed silhouette instead of cutting a
   # hard rectangular hole into an adjacent wall reconstruction.
   keep=cv2.dilate(masks[other],np.ones((21,21),np.uint8))
   distance=cv2.distanceTransform(255-keep,cv2.DIST_L2,5)
   protection=np.clip(1-distance/60,0,1)
   protection=protection*protection*(3-2*protection)
   blend*=1-protection
  blend[rail_core]=0
  # Do not wash out the floor, baseboard, or furniture outside the wall.
  blend[int(h*.84):]=0
  base=np.uint8(base*(1-blend[:,:,None])+model*blend[:,:,None])
  # Rebuild rail pixels under each removed item from unoccluded rail pixels.
  for row in np.where(rail_core)[0]:
   if row<y-24 or row>b+10:continue
   donor=rgb[row,outside[row]&(xx[row]>w*.15)&(xx[row]<w*.90)]
   if len(donor)>20:
    left=max(0,x-12);right=min(w,r+12)
    # Smooth row colour preserves rail borders without duplicating screws.
    tone=np.median(donor,axis=0)
    edge=np.minimum(np.arange(right-left)+1,np.arange(right-left,0,-1))/6
    edge=np.clip(edge,0,1)[:,None]
    base[row,left:right]=rgb[row,left:right]*(1-edge)+tone*edge
 # Preserve complete furnishing boundaries, including their antialiased edges.
 n,lab,stats,_=cv2.connectedComponentsWithStats(np.uint8(brown&outside))
 for a,c,d,e,area in stats[1:]:
  if d>w*.025 and e>h*.045 and area>150:
   region=np.zeros((h,w),np.uint8)
   region[max(0,c-4):min(h,c+e+4),max(0,a-4):min(w,a+d+4)]=255
   silhouette=np.uint8(np.linalg.norm(rgb.astype(float)-model.astype(float),axis=2)>25)*region
   preserve=cv2.GaussianBlur(cv2.dilate(silhouette,np.ones((5,5),np.uint8)),(5,5),1).astype(float)/255
   # Furniture preservation must never repaint removed hats/straps along
   # the rail. The rail has already been reconstructed from clean row donors.
   preserve[removed]=0
   base=np.uint8(base*(1-preserve[:,:,None])+rgb*preserve[:,:,None])
 # Audit the FINAL plate (after furniture preservation), never an intermediate.
 # All surviving items must be substantially changed and match the fitted wall.
 # Furniture-crossing objects are rejected above rather than erased as wall.
 failures={};metrics={}
 for iid,m in masks.items():
  core=cv2.erode(m,np.ones((7,7),np.uint8))>0
  if not core.any():core=m>0
  change=np.linalg.norm(base.astype(float)-rgb.astype(float),axis=2)
  residual=np.linalg.norm(base.astype(float)-model.astype(float),axis=2)
  changed=float(np.mean(change[core]>24)) if core.any() else 0
  background=float(np.mean(residual[core]<24)) if core.any() else 0
  metrics[iid]={'changed_fraction':round(changed,4),'background_fraction':round(background,4),'audited':not bool(reasons[iid])}
  if not reasons[iid]:
   why=[]
   if changed<.80:why.append('plate removal check: unchanged foreground')
   if background<.95:why.append('plate background check: residual foreground')
   x,y,r,b=boxes[iid]
   # Detect sharp reconstruction boundaries only on originally smooth wall,
   # excluding real object/wood edges. Wide feathers keep these below 8 RGB.
   error=base.astype(float)-model.astype(float)
   jump=np.maximum(np.linalg.norm(np.diff(error,axis=0,prepend=error[:1]),axis=2),np.linalg.norm(np.diff(error,axis=1,prepend=error[:,:1]),axis=2))
   smooth_wall=cv2.erode(np.uint8(np.linalg.norm(rgb.astype(float)-model.astype(float),axis=2)<24),np.ones((5,5),np.uint8))>0
   region=np.zeros((h,w),bool);region[max(0,y-100):min(h,b+110),max(0,x-100):min(w,r+110)]=True
   samples=smooth_wall&region
   seam_fraction=float(np.mean(jump[samples]>8)) if samples.any() else 0
   metrics[iid]['seam_fraction']=round(seam_fraction,5)
   if seam_fraction>.02:why.append('patch seam check: abrupt wall transition')
   x,y,r,b=boxes[iid];rail_errors=[]
   for row in np.where(rail_core)[0]:
    if row<y-24 or row>b+10:continue
    donor=rgb[row,outside[row]&(xx[row]>w*.15)&(xx[row]<w*.90)]
    if len(donor)>20:
     rail_errors.extend((np.linalg.norm(base[row,x:r].astype(float)-np.median(donor,axis=0),axis=1)>32).tolist())
   rail_background=1-float(np.mean(rail_errors)) if rail_errors else 1
   metrics[iid]['rail_background_fraction']=round(rail_background,4)
   if rail_background<.93:why.append('rail background check: silhouette fragments')
   # Source boxes which omit handles above the silhouette cannot be certified.
   x,y,r,b=boxes[iid]
   if zid.startswith('bags-') and y<h*.20 and np.any(rail_core[:y+1]):
    why.append('upper strap outside extraction: incomplete silhouette')
   if why:failures[iid]=why
 if failures:
  return reconstruct(rgb,boxes,zid,{**rejected,**failures})
 (REVIEW/f'{zid}-integrity.json').write_text(json.dumps(metrics,indent=2)+'\n')
 return base,masks,reasons

def review(zid,base,rgb,masks,reasons):
 im=Image.fromarray(base);im.thumbnail((1200,1200));im.save(REVIEW/f'{zid}-plate.jpg',quality=90)
 sheet=Image.new('RGB',(1000,240*((len(masks)+3)//4)), 'white');draw=ImageDraw.Draw(sheet)
 for j,(iid,m) in enumerate(masks.items()):
  y,x=np.where(m>0)
  if not len(x):continue
  box=(x.min(),y.min(),x.max()+1,y.max()+1);rgba=Image.fromarray(np.dstack([rgb,m])).crop(box);rgba.thumbnail((230,185))
  tile=np.indices((190,240)).sum(0)//12%2*25+210;tile=Image.fromarray(np.repeat(tile[:,:,None],3,2).astype('uint8'))
  tile.paste(rgba,((240-rgba.width)//2,0),rgba);px=j%4*250;py=j//4*240;sheet.paste(tile,(px,py));draw.text((px,py+190),iid+' '+('FAIL' if reasons[iid] else 'PASS'),fill='black');draw.text((px,py+207),'; '.join(reasons[iid])[:34],fill='black')
 sheet.save(REVIEW/f'{zid}-layers.jpg',quality=90)

def fix_card(im,iid,config,paint=None,box=None):
 if iid not in config:return im
 if config[iid]['method'] in ('upscale','cuff-clone','hook','fringe'):
  import sys
  sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data/source'))
  from overrides_v21 import card
  return card(im,config[iid])
 a=np.array(im);alpha=a[:,:,3];rgb=a[:,:,:3]
 if config[iid]['method']=='opaque-pink':
  mask=np.uint8(alpha>8)*255
  contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);mask[:]=0;cv2.drawContours(mask,contours,-1,255,-1)
  alpha=cv2.GaussianBlur(mask,(3,3),.65)
  x,y,r,b=box;sample=paint[y:b,x:r];sel=(sample[:,:,0]>sample[:,:,1]*1.35)&(sample[:,:,2]>sample[:,:,1]*1.1)
  target=np.median(sample[sel],axis=0) if sel.any() else np.array([218,57,151])
  lum=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float);v=lum[mask>0];shade=np.clip((lum-np.median(v))/max(15,np.std(v))*.16+1,.5,1.4)
  rgb=np.clip(target[None,None,:]*shade[:,:,None],0,255).astype('uint8')
  # Retain original neutral metal hook and wooden hanger above the neckline.
  foreground_rows=np.where((mask>0).sum(1)>mask.shape[1]*.30)[0]
  if len(foreground_rows):
   top=int(foreground_rows[0]);rgb[:top]=a[:top,:,:3]
 else:
  core=alpha>220;nearest=cv2.inpaint(rgb,np.uint8((alpha>0)&~core)*255,3,cv2.INPAINT_TELEA);rgb=np.where(((alpha>0)&(alpha<180))[:,:,None],nearest,rgb)
 return Image.fromarray(np.dstack([rgb,alpha]))
