"""Deterministic v2.1 wall reconstruction and documented card overrides."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
REVIEW=Path('/workspace/collection-room/work/v2.1/review')
REVIEW.mkdir(parents=True,exist_ok=True)

def bag_wall_bottom(rgb, boxes, hanging=False):
 """Strong horizontal wall/baseboard edge, sampled away from bag columns."""
 h,w=rgb.shape[:2]
 gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float)
 # Side margins are clean in the bag room paintings; median rejects a leg.
 cols=np.r_[int(w*.035):int(w*.09),int(w*.92):int(w*.97)]
 profile=np.median(gray[:,cols],axis=1)
 edges=np.abs(np.diff(profile))
 start=int(h*.76) if hanging else max(int(h*.80), min(int(h*.94),max(b for x,y,r,b in boxes.values())+8))
 candidates=np.where(edges[start:int(h*.96)]>8)[0]
 return start+int(candidates[0])+1 if len(candidates) else int(h*.86)



def hat_wall_boundary(rgb, boxes):
 """Fit the main wall's baseboard edge from clean columns on its right side.
 Opposite side walls can have a different baseboard height; pooling their
 row medians let wall reconstruction eat a ten-pixel strip of the main trim.
 """
 h,w=rgb.shape[:2]
 gray=cv2.GaussianBlur(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float),(5,1),1)
 edges=np.abs(np.diff(gray,axis=0));points=[]
 for x in range(round(w*.40),round(w*.98),8):
  candidates=np.where(edges[round(h*.76):round(h*.92),x]>12)[0]+round(h*.76)
  for y in candidates:
   if not any(a-8<=x<d+8 and c-8<=y<e+8 for a,c,d,e in boxes.values()):
    points.append((x,y));break
 if len(points)>=20:
  pts=np.array(points,float);keep=np.ones(len(pts),bool)
  for _ in range(4):
   slope,offset=np.polyfit(pts[keep,0],pts[keep,1],1)
   residual=pts[:,1]-(slope*pts[:,0]+offset)
   keep=np.abs(residual-np.median(residual[keep]))<4
   if keep.sum()<20:break
  if keep.sum()>=20 and abs(slope)<.10:
   return np.clip(np.floor(slope*np.arange(w)+offset)-1,0,h).astype(int)
 return np.full(w,bag_wall_bottom(rgb,boxes,hanging=True),int)


def bag_hooks(rgb, boxes, rail_core, model):
 """Keep thin neutral metal stems crossing a wooden rail on the plate.
 Find the stem above the wood, then follow its narrow column through the rail;
 a padded item crop can contain a previous bag and is not a hook detector.
 """
 h,w=rgb.shape[:2]; result=np.zeros((h,w),np.uint8)
 runs=np.split(np.where(rail_core)[0],np.where(np.diff(np.where(rail_core)[0])>1)[0]+1)
 chroma=np.ptp(rgb.astype(int),axis=2)
 gray=rgb.mean(2).astype(np.float32)
 stem_edge=gray-cv2.GaussianBlur(gray,(21,1),4)<-12
 for x,y,r,b in boxes.values():
  candidates=[a for a in runs if len(a)>4 and a[0]<y+15 and -h*.04<=y-a[-1]<h*.30]
  if not candidates:continue
  rows=min(candidates,key=lambda a:abs(y-a[-1]));rt,rb=int(rows[0]),int(rows[-1])
  mid=(x+r)//2; radius=max(15,int((r-x)*.13))
  left=max(0,mid-radius);right=min(w,mid+radius)
  top=max(0,rt-12)
  seed_top=max(top,rt-7)
  seed=(chroma[seed_top:rt-1,left:right]<35)&stem_edge[seed_top:rt-1,left:right]
  seed=cv2.morphologyEx(np.uint8(seed),cv2.MORPH_CLOSE,np.ones((1,5),np.uint8))
  nn,ll,ss,_=cv2.connectedComponentsWithStats(np.uint8(seed))
  opts=[]
  for k in range(1,nn):
   a,c,d,e,area=ss[k]
   if e>=2 and d<=16 and area>=5:opts.append((abs(left+a+d/2-mid),left+a+d/2))
  if not opts:
   # Some paintings put the entire hook inside the rail, with no stem
   # above its top edge. Neutral bright metal is distinct from warm wood.
   seed=(chroma[rt:rb+1,left:right]<20)&(gray[rt:rb+1,left:right]>100)
   nn,ll,ss,_=cv2.connectedComponentsWithStats(np.uint8(seed))
   for k in range(1,nn):
    a,c,d,e,area=ss[k]
    if 2<=d<=18 and e>max(8,d*1.4) and area>=15:
     opts.append((abs(left+a+d/2-mid),left+a+d/2))
  if not opts:continue
  hx=round(min(opts)[1]);l=max(0,hx-5);rgt=min(w,hx+6);bottom=min(h,rb+9)
  # The narrow neutral metal is distinct from warm wood/canvas and dark cloth.
  metal=(chroma[top:bottom,l:rgt]<35)&(rgb[top:bottom,l:rgt].mean(2)>85)
  # Include the dark outline only alongside a bright metal pixel.
  bright=cv2.dilate(np.uint8(metal),np.ones((3,7),np.uint8))>0
  metal=bright&(chroma[top:bottom,l:rgt]<45)
  flank=np.r_[max(0,hx-18):max(0,hx-10),min(w,hx+10):min(w,hx+18)]
  side=np.median(rgb[top:bottom,flank],axis=1)
  contrast=np.linalg.norm(rgb[top:bottom,l:rgt].astype(float)-side[:,None,:],axis=2)>25
  metal[:rt-top]&=contrast[:rt-top]
  metal[rb+1-top:]&=contrast[rb+1-top:]
  # Below the curved head, follow the bright shaft rather than adjacent
  # handle highlights. Neutral rail shadows are broad, not metal stems.
  sample=rgb[rt+5:min(rb,rt+17),l:rgt].mean(2)
  if len(sample):
   neutral=chroma[rt+5:min(rb,rt+17),l:rgt]<30
   shaft=int(np.argmax(np.sum(sample*neutral,axis=0)))
   cols=np.arange(rgt-l)
   metal[max(0,rt+8-top):]&=np.abs(cols-shaft)[None,:]<=5
  below=max(0,rb+1-top)
  # Keep the shaft's dark outline together with its bright centre. A strict
  # brightness cut left a one-pixel metal stub swinging with the handle.
  shaft_light=rgb[rb+1:bottom,l:rgt].mean(2)>model[rb+1:bottom,l:rgt].mean(2)*.85
  shaft_light=cv2.dilate(np.uint8(shaft_light),np.ones((3,3),np.uint8))>0
  metal[below:] &= shaft_light
  result[top:bottom,l:rgt]|=np.uint8(metal)*255
 return result


def attached_wall_fraction(rgb, plate, mask):
 """Measure coherent wall patches, not chance colour matches in woven cloth.
 White crinkles can match wall colour along one-pixel chains reaching the
 silhouette edge. A 3x3 opening breaks those chains; actual wall wedges and
 filled handle openings retain their area and remain subject to the 3% gate.
 """
 wallish=np.uint8((np.linalg.norm(rgb.astype(float)-plate.astype(float),axis=2)<14)&(mask>0))
 wallish=cv2.morphologyEx(wallish,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
 _,labels=cv2.connectedComponents(wallish)
 edge=(mask>0)&(cv2.erode(mask,np.ones((3,3),np.uint8))==0)
 touching=np.unique(labels[edge&(wallish>0)]);touching=touching[touching>0]
 return float(np.isin(labels,touching).sum()/max(1,np.count_nonzero(mask)))


def configured_rails(rgb, outside, specs, shadow_depth=58):
 """Sample straight perspective rails in rail coordinates, not image rows.
 Geometry is zone-level; every depth has independent unoccluded wood donors.
 """
 h,w=rgb.shape[:2]; yy,xx=np.indices((h,w))
 region=np.zeros((h,w),bool); paint=rgb.copy(); supported=True
 for spec in specs:
  left,top,right,end,bottom,endbottom=spec
  left*=w/1536;right*=w/1536;top*=h/1024;end*=h/1024;bottom*=h/1024;endbottom*=h/1024
  t=(np.arange(w)-left)/(right-left)
  upper=top+(end-top)*t;lower=bottom+(endbottom-bottom)*t
  depth=np.rint((yy-upper[None,:])/np.maximum(1,lower-upper)[None,:]*48).astype(int)
  rr=(xx>left+5)&(xx<right-5)&(depth>=-3)&(depth<=shadow_depth)
  for k in range(-3,shadow_depth+1):
   target=rr&(depth==k)
   # Thin level rails skip quantized depths entirely; absent pixels need no
   # donors. Every depth actually painted still requires clean samples.
   if not target.any():continue
   donor=target&outside
   if np.count_nonzero(donor)<20:supported=False;continue
   paint[target]=np.median(rgb[donor],axis=0)
  region|=rr
 return region,paint,supported


def configured_surfaces(rgb, model, specs, outside=None):
 """Rebuild finite straight surfaces from clean columns at the same height.
 Bboxes/polygons describe furniture geometry, not individual item silhouettes.
 An unsupported donor row is never certified for removal.
 """
 h,w=rgb.shape[:2]; model=model.copy()
 surface=np.zeros((h,w),bool); supported=np.zeros((h,w),bool)
 for spec in specs:
  a,c,d,e=spec['bbox'];a=round(a*w/1536);d=round(d*w/1536);c=round(c*h/1024);e=round(e*h/1024)
  donor_columns=np.concatenate([np.arange(round(l*w/1536),round(r*w/1536)) for l,r in spec['donor_x']])
  donor_columns=donor_columns[(donor_columns>=0)&(donor_columns<w)]
  region=np.zeros((h,w),np.uint8)
  if spec.get('polygon'):
   points=np.array([[round(x*w/1536),round(y*h/1024)] for x,y in spec['polygon']],np.int32)
   cv2.fillPoly(region,[points],1)
   region[:c]=0;region[e:]=0;region[:,:a]=0;region[:,d:]=0
  else:region[c:e,a:d]=1
  for row in range(c,e):
   donors=donor_columns if outside is None else donor_columns[outside[row,donor_columns]]
   if len(donors)<8:continue
   target=region[row]>0
   model[row,target]=np.median(rgb[row,donors],axis=0)
   supported[row,target]=True
  surface|=region.astype(bool)
 return model,surface,supported


def furnishing_hooks(rgb, model, masks, boxes, rail_geometry):
 """Follow thin neutral metal from the rail, stopping at substantial textile.
 A bbox-centre stem is insufficient for paired or diagonal hooks, and can hit
 the item in the row above. Material and connectivity determine ownership.
 """
 h,w=rgb.shape[:2]; gray=rgb.mean(2).astype(np.float32)
 wall_match=np.linalg.norm(rgb.astype(float)-model.astype(float),axis=2)<30
 solid=np.zeros((h,w),np.uint8)
 for m in masks.values():
  body=m.copy();body[wall_match]=0
  body=cv2.morphologyEx(body,cv2.MORPH_OPEN,np.ones((11,11),np.uint8))
  solid|=cv2.dilate(body,np.ones((3,3),np.uint8))
 neutral=(np.ptp(rgb.astype(int),axis=2)<35)&(gray>20)
 contrast=np.abs(gray-cv2.GaussianBlur(gray,(21,1),4))>9
 candidate=np.uint8(neutral&contrast&(solid==0))
 candidate=cv2.morphologyEx(candidate,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
 candidate[solid>0]=0
 nn,ll,ss,_=cv2.connectedComponentsWithStats(candidate)
 result=np.zeros((h,w),np.uint8)
 for k in range(1,nn):
  a,c,d,e,area=ss[k]
  if not (e>=8 and d<max(45,w*.04) and e<h*.16 and area<1600):continue
  component=ll[c:c+e,a:a+d]==k
  attached=np.any(component&rail_geometry[c:c+e,a:a+d])
  # Small independent wall hooks sometimes support a second scarf tip.
  detached=False
  near_top=any(x-15<=a<r+15 and y-20<=c<y+(b-y)*.10 for x,y,r,b in boxes.values())
  values=gray[c:c+e,a:a+d][component]
  outlined=values.min()<70 and values.max()>160
  if d<22 and area<350 and near_top and outlined:
   # A wall hook rises above its local textile boundary. Neutral scarf side
   # edges can also be thin, but already have solid textile beside their top.
   local=solid[c:min(h,c+e+30),max(0,a-8):min(w,a+d+8)]
   rows=np.where(local.any(1))[0]
   detached=bool(len(rows) and rows[0]>=8)
  if attached or detached:
   keep=ll==k
   if detached and not attached:keep[c+int(rows[0])+3:]=False
   result[keep]=255
 result=cv2.dilate(result,np.ones((3,3),np.uint8));result[(solid>0)|~neutral]=0
 return result


def reconstruct(rgb, boxes, zid, rejected=None):
 rejected = rejected or {}
 h,w=rgb.shape[:2]; outside=np.ones((h,w),bool)
 config_path=Path(__file__).resolve().parents[1]/'data/source/layers-v23.json'
 config=json.loads(config_path.read_text()).get(zid,{}) if config_path.exists() else {}
 if config.get('garment_rack'):
  from v23_garments import reconstruct as reconstruct_garments
  return reconstruct_garments(rgb, boxes, config['garment_rack'])
 wall_bags=zid.startswith('bags-') and not config.get('surfaces')
 wall_boundary=hat_wall_boundary(rgb,boxes) if zid.startswith('hats-') or config.get('surface_reconstruction') else None
 for x,y,r,b in boxes.values(): outside[max(0,y-5):min(h,b+5),max(0,x-5):min(w,r+5)]=False
 # Exclude contact-shadow donors as well as the item itself.
 # Neutral wall samples exclude windows, rails, floor and coloured furnishings.
 wall_samples=outside.copy()
 for x,y,r,b in boxes.values():
  wall_samples[max(0,y-25):min(h,b+90),max(0,x-35):min(w,r+70)]=False
 # Dense accessory layouts leave too little support for the wide exclusion.
 neutral=(rgb.max(2).astype(float)-rgb.min(2)<22)&(rgb.mean(2)>125)&wall_samples
 yy,xx=np.indices((h,w)); valid=neutral&(xx>w*.12)&(xx<w*.94)&(yy<h*.78)
 original_valid=(rgb.max(2).astype(float)-rgb.min(2)<22)&(rgb.mean(2)>125)&outside&(xx>w*.12)&(xx<w*.94)&(yy<h*.85)
 if valid.sum()<original_valid.sum()*.25 and not zid.startswith('bags-'):valid=original_valid
 # Dense two-row bag layouts can leave donors only above the first rail.
 # Extrapolating a quadratic from that sliver produces coloured/dark bands.
 # Use clean side wall as well when the donor rows cannot support the fit;
 # retain the wide item/shadow exclusions rather than training on handles.
 if wall_bags and (not valid.any() or np.ptp(yy[valid]) < h*.45):
  valid=neutral&(xx>w*.06)&(xx<w*.98)&(yy<h*.85)
 model=np.zeros_like(rgb,dtype=float)
 for ch in range(3):
  A=np.column_stack([np.ones(valid.sum()),xx[valid]/w,yy[valid]/h,(xx[valid]/w)**2,(yy[valid]/h)**2])
  if config.get('wall_fit')=='linear-y':A[:,4]=0  # sparse lower-wall donors beside long cords
  vals=rgb[:,:,ch][valid].astype(float)
  for _ in range(3):
   coeff=np.linalg.lstsq(A,vals,rcond=None)[0]; keep=np.abs(A@coeff-vals)<12;A=A[keep];vals=vals[keep]
  model[:,:,ch]=coeff[0]+coeff[1]*xx/w+coeff[2]*yy/h+coeff[3]*(xx/w)**2+coeff[4]*(yy/h)**2
 model=np.clip(model,0,255).astype(np.uint8)
 if wall_bags or zid.startswith('hats-') or config.get('surface_reconstruction'):
  # A low-order wall fit cannot represent a window's light pattern. Interpolate
  # its neutral positive residual from clean wall donors, never from cloth or
  # cast shadows. This keeps the light continuous through a removed silhouette.
  residual=rgb.astype(float)-model.astype(float)
  illumination=np.clip(residual.min(2),0,120 if zid.startswith('hats-') else 70).astype(np.uint8)
  wall_limit=wall_boundary[None,:] if wall_boundary is not None else h*.85
  unknown=~outside | (np.ptp(residual,axis=2)>15) | (yy>wall_limit)
  illumination[yy>wall_limit]=0
  small=(max(1,w//4),max(1,h//4))
  field=cv2.resize(illumination,small,interpolation=cv2.INTER_AREA)
  missing=cv2.resize(np.uint8(unknown)*255,small,interpolation=cv2.INTER_NEAREST)
  field=cv2.inpaint(field,missing,5,cv2.INPAINT_TELEA)
  field=cv2.GaussianBlur(cv2.resize(field,(w,h)),(0,0),8)
  field[yy>wall_limit]=0
  model=np.clip(model.astype(float)+field[:,:,None],0,255).astype(np.uint8)
 surface=np.zeros((h,w),bool);surface_supported=surface.copy()
 if config.get('surface_reconstruction'):
  # Surface colour must participate in segmentation. Fitting it only after
  # extraction drags bench fronts into a hanging layer as rectangular stubs.
  model,surface,surface_supported=configured_surfaces(rgb,model,config.get('surfaces',[]),outside)
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
 rail_geometry,rail_paint,geometry_valid=configured_rails(rgb,outside,config.get('rails',[]),96 if config.get('surface_reconstruction') else 58)
 # Rail rows are not background everywhere: crowns and pom-poms cross them.
 # Compare against clean row donors instead of deleting the whole +/-8 band.
 delta=np.linalg.norm(rgb.astype(float)-model.astype(float),axis=2)
 rail_tone=model[:,0].astype(float).copy()
 for row in np.where(rail)[0]:
  donor=rgb[row,outside[row]&(xx[row]>w*.15)&(xx[row]<w*.90)]
  if len(donor)>20:rail_tone[row]=np.median(donor,axis=0)
 # The dilated search band includes wall/shadow rows; rebuild only actual wood.
 # Bench test keeps the v2.1 row band; the wood-only core below is for repainting.
 bench_core=rail_core.copy()
 rail_core=rail&(rail_tone.mean(1)<195)
 if wall_bags:
  # A dark neutral cast-shadow row is not wood. Preserving it as a rail
  # produced hard rectangular remnants immediately above the lower rail.
  rail_core &= (rail_tone[:,0]-rail_tone[:,2]>15)
 # A rail is a thin horizontal wood band with independent clean donors.
 # Lower wall rails used to be labelled benches solely because of their y.
 # Keep furniture rejection for thick/unsupported bands, and audit repainted
 # rail pixels against these same donors below.
 validated_rail=np.zeros(h,bool)
 runs=np.split(np.where(rail_core)[0],np.where(np.diff(np.where(rail_core)[0])>1)[0]+1)
 for rows in runs:
  if not len(rows) or len(rows)>h*.09:continue
  supported=[row for row in rows if np.count_nonzero(brown[row]&outside[row])>max(20,w*.08)]
  if len(supported)>=len(rows)*.65:
   validated_rail[rows]=True
 # Black coated straps can be warm under room light; a strict neutral-only
 # test confuses their shaded buckles with the wooden rail.
 cloth_chroma=35 if wall_bags else 18
 dark_cloth=(rgb.mean(2)<90)&((rgb.max(2).astype(int)-rgb.min(2))<cloth_chroma)
 wood=rail[:,None]&(np.linalg.norm(rgb.astype(float)-rail_tone[:,None,:],axis=2)<35)&~dark_cloth
 if zid.startswith('hats-'):
  wood &= np.linalg.norm(rgb.astype(float)-rail_tone[:,None,:],axis=2)<18
  # Brown/olive corduroy may match rail luminance, but lacks the wood's
  # red-green chroma. A broad RGB radius alone punched holes in crown edges.
  wood &= np.abs((rgb[:,:,0].astype(float)-rgb[:,:,1])-(rail_tone[:,0]-rail_tone[:,1])[:,None])<6
 if rail_geometry.any():
  wood=rail_geometry&(np.linalg.norm(rgb.astype(float)-rail_paint.astype(float),axis=2)<22)&~dark_cloth
 if config.get('surface_reconstruction'):
  # The neutral rail shadow is plate geometry too, but is not wood: matching
  # it by colour must not punch horizontal holes in a grey quilted scarf.
  physical_rail=rail_geometry&(rail_paint[:,:,0].astype(int)-rail_paint[:,:,2]>15)
  wood&=~rail_geometry|physical_rail
  # Continue source wood grain through an occlusion with clean same-row
  # high-frequency donors; a flat median alone makes a visible rail patch.
  grain=rgb.astype(float)-cv2.GaussianBlur(rgb,(31,1),5).astype(float)
  clean=cv2.erode(np.uint8(outside),np.ones((1,31),np.uint8))>0
  for row in np.where(physical_rail.any(1))[0]:
   donors=np.where(physical_rail[row]&clean[row]&(np.ptp(rgb[row].astype(int),axis=1)>35))[0]
   if len(donors)<20:continue
   targets=np.where(physical_rail[row])[0]
   detail=grain[row,donors[(targets-targets[0])%len(donors)]]*.6
   rail_paint[row,targets]=np.uint8(np.clip(rail_paint[row,targets].astype(float)+detail,0,255))
  wood|=surface_supported&(np.linalg.norm(rgb.astype(float)-model.astype(float),axis=2)<35)
 rail_area=rail_geometry if rail_geometry.any() else np.broadcast_to(rail[:,None],(h,w))
 if config.get('surface_reconstruction') and rail_geometry.any():rail_area=physical_rail
 # Fine knit/embroidery texture is foreground evidence even when pale cloth
 # shares wall colour. A neutral-shadow seed alone erased an entire grey knit.
 gray=rgb.mean(2).astype(np.float32)
 texture=np.sqrt(np.maximum(0,cv2.blur(gray*gray,(5,5))-cv2.blur(gray,(5,5))**2))
 # Neutral, smooth darkening is cast shadow, not a textile foreground seed.
 ratio=(rgb.astype(float)+1)/(model.astype(float)+1)
 shadow_seed=(ratio.max(2)<.98)&(ratio.min(2)>.55)&(np.ptp(ratio,axis=2)<.10)&(rgb.mean(2)>100)
 hooks=bag_hooks(rgb,boxes,rail_core,model) if zid.startswith('bags-') else np.zeros((h,w),np.uint8)
 masks={};reasons={};extraction_top={}
 for iid,(x,y,r,b) in boxes.items():
  # Layout boxes locate items; they must not be hard extraction boundaries.
  pad=max(20,round(max(r-x,b-y)*.12))
  px=max(0,x-pad);py=max(0,y-pad);pr=min(w,r+pad);pb=min(h,b+pad)
  if zid.startswith('bags-'):
   # Layouts often describe only the body. Search to the nearest rail above
   # this item, rather than clipping the strap to the layout/padding boundary.
   above=np.where(rail[:y+1])[0]
   if len(above) and y-above[-1]<h*.30:
    py=min(py,max(0,int(above[-1])-12))
  extraction_top[iid]=py
  m=np.zeros((h,w),np.uint8)
  m[py:pb,px:pr]=np.uint8((delta[py:pb,px:pr]>24)&~wood[py:pb,px:pr])*255
  # Padding starts as probable background, so shadows do not train the cloth GMM.
  m[py:y]=0;m[b:pb]=0;m[:,:x]=0;m[:,r:]=0
  # Exclude neighbour cores before connected-component selection.
  for other,(a,c,d,e) in boxes.items():
   if other==iid:continue
   dx=int((d-a)*.18);dy=int((e-c)*.18);m[c+dy:e-dy,a+dx:d-dx]=0
  # Refine on a padded crop: central foreground seeds distinguish neutral cloth
  # from the adjacent neutral wall, while the wall model supplies probable seeds.
  local=np.where(m[py:pb,px:pr]>0,cv2.GC_PR_FGD,cv2.GC_PR_BGD).astype('uint8')
  if zid.startswith('hats-'):local[shadow_seed[py:pb,px:pr]]=cv2.GC_PR_BGD
  local[wood[py:pb,px:pr]]=cv2.GC_BGD
  if zid.startswith('hats-'):
   textile=(texture[py:pb,px:pr]>7)&(delta[py:pb,px:pr]>10)&~rail[py:pb,None]
   textile &= (xx[py:pb,px:pr]>x+(r-x)*.2)&(xx[py:pb,px:pr]<r-(r-x)*.2)&(yy[py:pb,px:pr]>y+(b-y)*.15)&(yy[py:pb,px:pr]<b-(b-y)*.15)
   centre=np.median(rgb[y+(b-y)//3:b-(b-y)//5,x+(r-x)//4:r-(r-x)//4].reshape(-1,3),axis=0)
   if centre.mean()>145 and np.ptp(centre)<25:local[textile]=cv2.GC_FGD
  local[[0,-1],:]=0;local[:,[0,-1]]=0
  cx=(x+r)//2-px;cy=(y+b)//2-py
  local[max(0,cy-4):cy+5,max(0,cx-4):cx+5]=cv2.GC_FGD
  cv2.setRNGSeed(42)
  cv2.grabCut(rgb[py:pb,px:pr],local,None,np.zeros((1,65)),np.zeros((1,65)),2,cv2.GC_INIT_WITH_MASK)
  m[:]=0;m[py:pb,px:pr]=np.uint8((local==1)|(local==3))*255;m[wood]=0
  m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));m[wood]=0
  if zid.startswith('bags-'):
   # GrabCut labels the narrow shaded inner fold of pale handles as wall.
   # Recover only a thin colour-discontinuous fringe, never fill the opening.
   fringe=(cv2.dilate(m,np.ones((5,5),np.uint8))>0)&(delta>28)&~wood
   m[fringe]=255
   # Dark straps crossing wood can be labelled background by the rail GMM.
   # Recover nearby neutral textile before the thin rail-texture rejection.
   strap=(cv2.dilate(m,np.ones((17,17),np.uint8))>0)&dark_cloth&rail[:,None]&~wood
   if wall_bags:m[strap]=255
  if zid.startswith('hats-') or config.get('surface_reconstruction'):
   fringe=(cv2.dilate(m,np.ones((3,3),np.uint8))>0)&(delta>28)&~wood
   m[fringe]=255
  # Reject thin horizontal rail texture attached to a dark crown.
  opened=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((5,1),np.uint8))
  thin_rail=rail_area&(opened==0)
  if not zid.startswith('hats-'):thin_rail &= dark_cloth
  m[thin_rail]=0
  if zid.startswith('bags-'):
   # Cream canvas is warm; the wall visible through its handle is neutral.
   # GrabCut can merge that wall into a shaded pale handle. Recover the opening
   # by material colour above the first broad canvas row, keeping the inner fold.
   warm=(rgb[:,:,0].astype(int)-rgb[:,:,2].astype(int)>12)&(rgb.mean(2)>120)
   widths=(warm&(m>0)).sum(1)
   body_rows=np.where((np.arange(h)>=y)&(widths>(r-x)*.50))[0]
   fabric=np.median(rgb[y+int((b-y)*.4):b,x:r].reshape(-1,3),axis=0)
   if len(body_rows) and 12<fabric[0]-fabric[2]<65 and fabric[0]>=fabric[1] and fabric[1]-fabric[2]>5:
    bt=int(body_rows[0])
    neutral_gap=(np.abs(rgb[:,:,0].astype(int)-rgb[:,:,2].astype(int))<10)&(rgb.mean(2)>110)
    m[neutral_gap&(yy<bt)&(yy>=y)]=0
   # Dark/coloured totes can enclose shadowed neutral wall between several
   # crossed handles. GrabCut follows the enclosing cloth and fills those
   # openings. Locate the solid body by a horizontal run, not total row width
   # (separated straps must not count as a broad body). Preserve pale canvas.
   # Midtone neutral fabric can itself look like shadowed wall (grey
   # shopping bags); restrict this recovery to dark or chromatic material.
   # Handle material can differ from the body (black webbing on a white
   # shopping tote). Use upper textile evidence as well as body colour.
   upper=(m>0)&(yy<y+(b-y)*.4)&(yy>=y)
   dark_handles=np.count_nonzero(upper&(rgb.mean(2)<80))>max(30,np.count_nonzero(upper)*.35)
   if fabric.mean()<100 or np.ptp(fabric)>35 or dark_handles:
    solid=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((1,max(3,round((r-x)*.65))),np.uint8))
    broad=solid.any(1).astype(np.uint8)
    persistent=cv2.erode(broad[:,None],np.ones((max(3,round((b-y)*.10)),1),np.uint8),anchor=(0,0))[:,0]>0
    body_rows=np.where((np.arange(h)>=y)&persistent)[0]
    if len(body_rows):
     # White printed webbing is neutral too, but brighter than the wall.
     gap=(np.ptp(rgb.astype(int),axis=2)<12)&(rgb.mean(2)>115)&(rgb.mean(2)<model.mean(2)+3)
     m[gap&(yy<int(body_rows[0]))]=0
  if zid.startswith('bags-'):
   # Pale/sheer handles can surround wall with the same neutral hue as the
   # fabric. Keep textured cloth and thin rope; remove only broad coherent
   # wall-coloured regions above the persistent solid body.
   solid=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((1,max(3,round((r-x)*.65))),np.uint8))
   persistent=cv2.erode(np.uint8(solid.any(1))[:,None],np.ones((max(3,round((b-y)*.10)),1),np.uint8),anchor=(0,0))[:,0]>0
   body_rows=np.where((np.arange(h)>=y)&persistent)[0]
   if len(body_rows):
    gap=(np.ptp(rgb.astype(int),axis=2)<12)&(rgb.mean(2)>115)&(rgb.mean(2)<model.mean(2)+3)&(yy<int(body_rows[0]))&(m>0)
    broad_gap=cv2.morphologyEx(np.uint8(gap),cv2.MORPH_OPEN,np.ones((7,7),np.uint8))
    nn,ll,ss,_=cv2.connectedComponentsWithStats(broad_gap)
    for k in range(1,nn):
     a,c,d,e,area=ss[k]
     if d>(r-x)*.15 and e>(b-y)*.15 and area>max(100,(r-x)*(b-y)*.005):
      opening=(cv2.dilate(np.uint8(ll==k),np.ones((7,7),np.uint8))>0)&gap
      m[opening]=0
  n,labels,stats,centers=cv2.connectedComponentsWithStats(m)
  reason=[]
  if n>1:
   scores=stats[1:,4]/(1+((centers[1:,0]-(x+r)/2)/(r-x))**2+((centers[1:,1]-(y+b)/2)/(b-y))**2)
   best=1+np.argmax(scores);m=np.uint8(labels==best)*255
  # Thin hooks remain on the plate. Stop at the first substantial textile row;
  # retain the original pixels there even when rebuilding the rail underneath.
  body=m
  if zid.startswith('accessories-'):
   body=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3) if config.get('surface_reconstruction') else (11,11),np.uint8))
  minimum_width=max(16,(r-x)*.13) if config.get('surface_reconstruction') else 10
  rows=np.where((body>0).sum(1)>minimum_width)[0]
  if len(rows) and not zid.startswith('bags-') and not config.get('surface_reconstruction'):
   top=rows[0]
   hook=m.copy();hook[top:]=0
   # GrabCut can disconnect the narrow metal stem from the cloth component.
   mid=(x+r)//2
   hook[py:top,max(px,mid-7):min(pr,mid+8)]=np.uint8(
    (delta[py:top,max(px,mid-7):min(pr,mid+8)]>24)&
    ~wood[py:top,max(px,mid-7):min(pr,mid+8)])*255
   if zid.startswith('hats-'):
    above=np.where(rail_area[:top].any(1))[0]
    if len(above) and top-above[-1]<h*.15:
     end=int(above[-1]);start=end
     while start>0 and rail_area[start-1].any():start-=1
     hook[:max(py,start-12)]=0
     hook[np.ptp(rgb.astype(int),axis=2)>=30]=0
     hook[rgb.mean(2)<80]=0
    else:hook[:]=0
   hook=cv2.dilate(hook,np.ones((3,3),np.uint8));hook[top:]=0
   hooks|=hook
   if zid.startswith('hats-'):m[hooks>0]=0
   else:m[:top]=0
  if zid.startswith('accessories-') and not config.get('surface_reconstruction'):
   # Silver stems can continue a few pixels into a coiled belt. Retain whole
   # narrow metal components, rather than leaving their lower tips in the layer.
   metal=np.uint8((rgb.min(2)>85)&(rgb.max(2)<240)&
    (np.ptp(rgb.astype(int),axis=2)<35)&(m>0)&(yy<y+(b-y)*.22))
   nn,ll,ss,_=cv2.connectedComponentsWithStats(metal)
   for k in range(1,nn):
    a,c,d,e,area=ss[k]
    if e>max(8,d*2) and np.any((ll==k)&(cv2.dilate(hooks,np.ones((5,5),np.uint8))>0)) and d<max(20,(r-x)*.4) and area<(r-x)*(b-y)*.03 and abs(a+d/2-(x+r)/2)<(r-x)*.25:
     stem=cv2.dilate(np.uint8(ll==k),np.ones((3,3),np.uint8))>0
     hooks[stem]=255;m[stem]=0
  if zid.startswith('hats-'):
   textile_evidence=(texture>6)|(np.ptp(rgb.astype(int),axis=2)>25)|(rgb.mean(2)<110)
   widths=((m>0)&textile_evidence).sum(1)
   broad_rows=np.where((widths>(r-x)*.5)&(yy[:,0]>=y))[0]
   if len(broad_rows) and broad_rows[-1]<b-(b-y)*.12:
    bt=int(broad_rows[-1])
    gap=(m>0)&(yy>bt)&(np.ptp(rgb.astype(int),axis=2)<16)&(rgb.mean(2)>120)&(texture<6)
    clean_gap=cv2.morphologyEx(np.uint8(gap),cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    clean_gap=(cv2.dilate(clean_gap,np.ones((5,5),np.uint8))>0)&gap
    m[clean_gap]=0
  # Fill small enclosed textile holes, never a convex hull (diagonal wall wedges).
  # Retain genuine handle/strap gaps, including shadow-darkened wall inside them.
  inverse=np.uint8(m==0);nn,ll,ss,_=cv2.connectedComponentsWithStats(inverse)
  for k in range(1,nn):
   # Bag openings are structural, including small handles on short totes.
   hole_limit=.001 if zid.startswith('bags-') else .08
   enclosed=ss[k,4] < (r-x)*(b-y)*hole_limit and ss[k,0]>x and ss[k,1]>y and ss[k,0]+ss[k,2]<r and ss[k,1]+ss[k,3]<b
   if not enclosed:continue
   if zid.startswith('hats-'):
    # Embroidery holes lie within broad textile rows. The space between ear
    # flaps or long chin cords is wall, even when a cord closes that space.
    filled_rows=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((1,max(3,round((r-x)*.20))),np.uint8))
    broad=(filled_rows>0).sum(1)>(r-x)*.55
    textile_hole=np.all(broad[ss[k,1]:ss[k,1]+ss[k,3]])
    if textile_hole and not np.any(rail[ss[k,1]:ss[k,1]+ss[k,3]]):m[ll==k]=255
   elif np.mean(shadow_seed[ll==k])<.5:m[ll==k]=255
  m[wood]=0
  if zid.startswith('hats-'):
   # Remove horizontal grain threads while retaining vertical cords. Closing
   # tiny holes only after wood rejection keeps woven crowns solid.
   tall=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((9,1),np.uint8))
   m[rail_area&(tall==0)&(np.abs(xx-(x+r)//2)>12)]=0
   # Vertical corduroy ribs continue through the rail band; horizontal wood
   # grain does not. Recover those near-crown pixels instead of leaving the
   # wood-colour test's narrow vertical slits in brown crowns.
   gx=cv2.blur(np.abs(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)),(3,3))
   gy=cv2.blur(np.abs(cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)),(3,3))
   ribs=rail_area&(gx>12)&(gx>gy*1.5)&(np.abs(xx-(x+r)/2)<(r-x)*.3)&(cv2.dilate(m,np.ones((9,1),np.uint8))>0)
   m[ribs]=255
   closed=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((3,5),np.uint8))
   m[rail_area]=closed[rail_area]
   inv=np.uint8(m==0);nn,ll,ss,_=cv2.connectedComponentsWithStats(inv)
   for k in range(1,nn):
    if ss[k,4]<(r-x)*(b-y)*.003 and ss[k,0]>x and ss[k,1]>y and ss[k,0]+ss[k,2]<r and ss[k,1]+ss[k,3]<b:m[ll==k]=255
  if zid.startswith(('bags-','hats-')):m[hooks>0]=0
  cov=np.count_nonzero(m)/max(1,(r-x)*(b-y))
  if not .15<cov<.98:reason.append('mask coverage out of range')
  foreground_y=np.where(m>0)[0]
  if zid.startswith('bags-') and len(foreground_y) and foreground_y.min()>y+(b-y)*.18:reason.append('missing pale handle or upper silhouette')
  # Height alone is not a surface test: long scarves can hang entirely on
  # wall below 83%. Use the detected main-wall/baseboard geometry instead.
  if zid.startswith('accessories-') and np.any((m>0)&(yy>=(wall_boundary[None,:] if wall_boundary is not None else h*.83))&~surface_supported):
   reason.append('floor or furnishing reconstruction unsupported')
  # A wall fit cannot remove objects crossing lower wood/furniture surfaces.
  unvalidated_wood=bench_core.copy()
  if zid.startswith('hats-'):unvalidated_wood &= ~cv2.dilate(np.uint8(validated_rail)[:,None],np.ones((17,1),np.uint8))[:,0].astype(bool)
  if zid.startswith('accessories-'):
   # Finite benches do not occupy their entire scanline. Clean wood donors
   # bound each furnishing row in x; a scarf beside the bench is still on wall.
   furnishing_rows=unvalidated_wood & (np.arange(h)>=h*.70)
   crossing=False
   for row in np.where(furnishing_rows)[0]:
    donors=np.where(brown[row]&outside[row])[0]
    if len(donors)>20 and np.any((m[row]>0)[max(0,donors.min()-4):min(w,donors.max()+5)] & ~surface_supported[row,max(0,donors.min()-4):min(w,donors.max()+5)]):crossing=True
   if crossing:reason.append('bench boundary: wood reconstruction unvalidated')
  elif not zid.startswith('bags-') and np.any(unvalidated_wood[max(y,int(h*.48)):b+8]):reason.append('bench boundary: wood reconstruction unvalidated')
  if config.get('rails') and not geometry_valid:reason.append('rail geometry: insufficient clean donors')
  reason.extend(rejected.get(iid, []))
  if (np.any(m[:2]) if zid.startswith('hats-') else y<h*.05):reason.append('item touches image boundary')
  masks[iid]=m;reasons[iid]=reason
 if config.get('surface_reconstruction'):
  hooks=furnishing_hooks(rgb,model,masks,boxes,physical_rail)
  near_hooks=cv2.dilate(hooks,np.ones((41,41),np.uint8))>0
  gap_delta=np.linalg.norm(rgb.astype(float)-model.astype(float),axis=2)
  shadow_match=rail_geometry&~physical_rail
  gap_delta[shadow_match]=np.minimum(gap_delta[shadow_match],np.linalg.norm(rgb.astype(float)-rail_paint.astype(float),axis=2)[shadow_match])
  gap=near_hooks&(np.ptp(rgb.astype(int),axis=2)<25)&(gap_delta<28)
  # A diagonal hook can enclose a smooth wedge of wall in the initial cloth
  # component. Remove coherent wall there, preserving textured pale yarn.
  hook_gap=cv2.morphologyEx(np.uint8(gap),cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
  hook_gap=(cv2.dilate(hook_gap,np.ones((3,3),np.uint8))>0)&gap
  for m in masks.values():m[(hooks>0)|hook_gap]=0
 # Long chin straps/tassels can cross the baseboard. Reconstruct only a
 # narrow silhouette neighbourhood there; a wall fit must not erase the floor.
 # Wide furnishing-crossing bodies still fail until a surface is configured.
 lower_repair=np.zeros((h,w),np.uint8);lower_alpha=np.zeros((h,w),float);lower_model=rgb.copy()
 if zid.startswith('hats-'):
  boundary=wall_boundary
  for iid,m in masks.items():
   lower=m.copy();lower[yy<boundary[None,:]-4]=0
   if not np.any(lower) or reasons[iid]:continue
   if np.any(cv2.morphologyEx(lower,cv2.MORPH_OPEN,np.ones((1,max(3,round(w*.06))),np.uint8))):
    reasons[iid].append('floor or furnishing: wide silhouette needs surface donors');continue
   sy,sx=np.where(lower>0);a=max(0,int(sx.min())-65);d=min(w,int(sx.max())+80)
   c=max(0,int(boundary[a:d].min())-12);e=min(h,int(sy.max())+56)
   donor_left=np.arange(max(0,a-60),max(1,a-25))
   donor_right=np.arange(min(w-1,d+25),min(w,d+60))
   # Straight baseboard/floor surfaces are sampled at the SAME y. Inpainting
   # across their edge bent it into tassel-shaped grey blobs. Interpolate
   # clean flank strips instead; remove contact shadows with a soft margin.
   ramp=np.linspace(0,1,d-a)[None,:,None]
   left=np.median(rgb[c:e,donor_left],axis=1)[:,None,:]
   right=np.median(rgb[c:e,donor_right],axis=1)[:,None,:]
   tones=left*(1-ramp)+right*ramp
   detail=rgb[c:e,a:d].astype(float)-cv2.GaussianBlur(rgb,(0,0),3)[c:e,a:d].astype(float)
   texture_region=(yy[c:e,a:d]>boundary[None,a:d]+8)&(cv2.dilate(m,np.ones((13,13),np.uint8))[c:e,a:d]==0)
   tones[texture_region]+=detail[texture_region]
   lower_model[c:e,a:d]=np.uint8(np.clip(tones,0,255))
   core=cv2.dilate(lower,np.ones((41,41),np.uint8));core[:c]=0;core[e:]=0
   distance=cv2.distanceTransform(255-core,cv2.DIST_L2,5)
   alpha=np.clip(1-distance/30,0,1);alpha=alpha*alpha*(3-2*alpha)
   region=np.zeros((h,w),bool);region[c:e,a:d]=True
   alpha*=np.clip((yy-c)/10,0,1)
   alpha[~region]=0
   lower_alpha=np.maximum(lower_alpha,alpha);lower_repair|=core
 # Rebuild straight furnishing surfaces from clean columns of that same
 # surface. Config records geometry/donor strips, never item-painted pixels.
 if not config.get('surface_reconstruction'):
  model,surface,surface_supported=configured_surfaces(rgb,model,config.get('surfaces',[]))
 # Matching wood belongs to the plate, including the bench top directly
 # below a standing pouch; it must never swing as a rectangular layer stub.
 if surface.any():
  furnishing=surface&(np.linalg.norm(rgb.astype(float)-model.astype(float),axis=2)<35)
  for m in masks.values():m[furnishing]=0
 # Furniture is a finite rectangle, not a rail spanning the entire room.
 rail_core[np.any(surface,axis=1)]=False
 # Actual mask collisions, never bbox intersection alone.
 keys=list(masks)
 for i,a in enumerate(keys):
  for b in keys[i+1:]:
   if np.count_nonzero((masks[a]>0)&(masks[b]>0))>12:
    reasons[a].append('touches neighbour mask');reasons[b].append('touches neighbour mask')
 # Repainting a whole bag bbox with median RGB donor rows changes intact
 # wood chroma at rest. Outside silhouettes lift neutral darkening to the
 # donor luminance, retaining source chroma and brighter grain highlights.
 replacement=model.copy()
 if surface.any():
  union=np.zeros((h,w),bool)
  for m in masks.values():union|=m>0
  gain=np.maximum(1,model.mean(2)/np.maximum(1,rgb.mean(2)))
  texture=np.clip(rgb.astype(float)*gain[:,:,None],0,255).astype(np.uint8)
  replacement[surface&~union]=texture[surface&~union]
 base=rgb.copy()
 removed=np.zeros((h,w),bool)
 for iid,m in masks.items():
  if reasons[iid]:continue
  x,y,r,b=boxes[iid]
  # Remove painted contact shadows as well as the foreground silhouette.
  hole=np.zeros((h,w),np.uint8)
  # Distance feather has a fully reconstructed core, with a broad, smooth
  # transition rather than a Gaussian-blurred rectangular patch.
  top=max(0,y-24)
  if zid.startswith('bags-') and np.any(m):top=min(top,max(0,int(np.where(m>0)[0].min())-6))
  shadow_pad=65 if zid.startswith('hats-') else 25
  hole[top:min(h,b+shadow_pad),max(0,x-12):min(w,r+12)]=255
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
  surface_blend=blend.copy() if config.get('surface_reconstruction') else None
  if rail_geometry.any():blend[rail_geometry]=0
  else:blend[rail_core]=0
  blend[hooks>0]=0
  # Do not wash out the floor, baseboard, or furniture outside the wall.
  if zid.startswith('bags-'):
   # Lower hanging bags are still on wall. Locate its baseboard boundary
   # from clean side columns instead of a fixed 84% cutoff through the bags.
   boundary=bag_wall_bottom(rgb, boxes)
   blend[boundary:]=0
  elif zid.startswith('hats-'):
   blend[yy>=wall_boundary[None,:]]=0
  elif wall_boundary is not None:blend[yy>=wall_boundary[None,:]]=0
  else:blend[int(h*.84):]=0
  if surface_blend is not None:blend[surface_supported]=surface_blend[surface_supported]
  if surface.any():
   # A finite bench has sloping end caps. Never smear wall over the wood
   # outside its configured top/front polygon (including its AA border).
   furniture_rows=np.any(surface,axis=1)
   start=int(np.where(furniture_rows)[0][0])
   caps=cv2.dilate(np.uint8(brown&~surface&(yy>=start-40)),np.ones((3,3),np.uint8))>0
   blend[caps&(m==0)&~surface]=0
  base=np.uint8(base*(1-blend[:,:,None])+replacement*blend[:,:,None])
  # Rebuild rail pixels under each removed item from unoccluded rail pixels.
  rail_hole=cv2.dilate(m,np.ones((7,7),np.uint8))
  for row in ([] if rail_geometry.any() else np.where(rail_core)[0]):
   if row<y-24 or row>b+10:continue
   donor=rgb[row,outside[row]&(xx[row]>w*.15)&(xx[row]<w*.90)]
   if len(donor)>20:
    left=max(0,x-12);right=min(w,r+12)
    # Smooth row colour preserves rail borders without duplicating screws.
    tone=np.median(donor,axis=0)
    edge=np.minimum(np.arange(right-left)+1,np.arange(right-left,0,-1))/24
    edge=np.clip(edge,0,1)[:,None]
    # Repaint occluded rail only. A bbox can extend past the physical rail
    # end; replacing its entire row manufactured a rectangular wooden stub.
    if zid.startswith(('bags-','hats-')):
     edge[rail_hole[row,left:right]==0]=0
    edge[hooks[row,left:right]>0]=0
    base[row,left:right]=rgb[row,left:right]*(1-edge)+tone*edge
  if rail_geometry.any():
   target=rail_geometry&(rail_hole>0)&(hooks==0)
   if config.get('surface_reconstruction'):
    opacity=np.clip(1-cv2.distanceTransform(255-m,cv2.DIST_L2,5)/4,0,1)
    a=opacity[target,None]
    base[target]=np.uint8(base[target]*(1-a)+rail_paint[target]*a)
   else:base[target]=rail_paint[target]
 if config.get('surface_reconstruction'):
  # Reconstruct the whole rail-shadow band from clean same-row residuals.
  # Keeping it untouched except under masks left tiny textile ghosts and
  # bbox-shaped changes in the otherwise continuous rail shadow.
  shadow_region=rail_geometry&~physical_rail
  shadow_bottom=np.max(np.where(shadow_region,yy,-1),axis=0)
  for row in np.where(shadow_region.any(1))[0]:
   donors=shadow_region[row]&outside[row]
   if np.count_nonzero(donors)<20:continue
   tone=np.median(rgb[row,donors].astype(float)-model[row,donors].astype(float),axis=0)
   target=shadow_region[row]&(hooks[row]==0)
   columns=np.where(shadow_region[row])[0]
   side=np.clip(np.minimum(xx[row]-columns[0],columns[-1]-xx[row])/10,0,1)
   fade=side*np.clip((shadow_bottom-row)/6,0,1)
   # The independent rail audit uses this same donor-derived light/shadow
   # expectation, rather than comparing a lit wall to a flat dark row median.
   paint=np.uint8(np.clip(model[row].astype(float)+tone*fade[:,None],0,255))
   rail_paint[row,shadow_region[row]]=paint[shadow_region[row]]
   base[row,target]=paint[target]
 if np.any(lower_repair):base=np.uint8(base*(1-lower_alpha[:,:,None])+lower_model*lower_alpha[:,:,None])
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
   preserve[removed|surface]=0
   base=np.uint8(base*(1-preserve[:,:,None])+rgb*preserve[:,:,None])
 base[hooks>0]=rgb[hooks>0]
 # Audit the FINAL plate (after furniture preservation), never an intermediate.
 # All surviving items must be substantially changed and match the fitted wall.
 # Furniture-crossing objects are rejected above rather than erased as wall.
 failures={};metrics={}
 for iid,m in masks.items():
  core=cv2.erode(m,np.ones((7,7),np.uint8))>0
  if not core.any():core=m>0
  change=np.linalg.norm(base.astype(float)-rgb.astype(float),axis=2)
  expected=model.astype(float).copy()
  expected[lower_repair>0]=lower_model[lower_repair>0]
  if rail_geometry.any():expected[rail_geometry]=rail_paint[rail_geometry]
  else:expected[rail_core]=np.broadcast_to(rail_tone[:,None,:],rgb.shape)[rail_core]
  residual=np.linalg.norm(base.astype(float)-expected,axis=2)
  # Pale canvas can already be within 24 RGB of clean wall. Only pixels
  # distinguishable from the expected surface can measure removal by change;
  # the independent residual audit still covers the entire eroded silhouette.
  distinguishable=core
  if zid.startswith(('bags-','hats-')):
   distinguishable=core&(np.linalg.norm(rgb.astype(float)-expected,axis=2)>24)
  changed=float(np.mean(change[distinguishable]>24)) if distinguishable.any() else 1
  background=float(np.mean(residual[core]<24)) if core.any() else 0
  metrics[iid]={'changed_fraction':round(changed,4),'background_fraction':round(background,4),'audited':not bool(reasons[iid])}
  if not reasons[iid]:
   why=[]
   if zid.startswith('bags-'):
    # A numerically exact rest composite can still carry wall inside a layer.
    # Audit edge-connected wall matches independently of the plate-removal test.
    wall_fraction=attached_wall_fraction(rgb,base,m)
    metrics[iid]['wall_in_layer_fraction']=round(wall_fraction,4)
    if wall_fraction>.03:why.append('layer check: background attached to silhouette')
    # Failed neighbours are protected during repaint. That protection must not
    # leave a narrow ghost along an otherwise removed bag's outer silhouette.
    edge_residual=float(np.mean(residual[m>0]>24)) if np.any(m) else 1
    metrics[iid]['full_silhouette_residual']=round(edge_residual,4)
    if edge_residual>.005:why.append('plate edge check: residual silhouette near protected neighbour')
   if changed<.80:why.append('plate removal check: unchanged foreground')
   if background<.95:why.append('plate background check: residual foreground')
   x,y,r,b=boxes[iid]
   # Detect sharp reconstruction boundaries only on originally smooth wall,
   # excluding real object/wood edges. Wide feathers keep these below 8 RGB.
   # On furnishing, the expected surface includes retained source grain;
   # comparing it to a flat wall/row model mistakes real wood grain for seams.
   error=base.astype(float)-replacement.astype(float)
   jump=np.maximum(np.linalg.norm(np.diff(error,axis=0,prepend=error[:1]),axis=2),np.linalg.norm(np.diff(error,axis=1,prepend=error[:,:1]),axis=2))
   smooth_wall=cv2.erode(np.uint8(np.linalg.norm(rgb.astype(float)-model.astype(float),axis=2)<24),np.ones((5,5),np.uint8))>0
   region=np.zeros((h,w),bool);region[max(0,y-100):min(h,b+110),max(0,x-100):min(w,r+110)]=True
   samples=smooth_wall&region
   if zid.startswith('hats-'):samples&=~rail[:,None]
   if config.get('surface_reconstruction'):
    # The wall seam audit must not count untouched rug grain, real furniture
    # borders, or rail edges merely because their RGB happens to fit the wall.
    samples&=(yy<wall_boundary[None,:])&~surface&~rail_geometry
   if zid.startswith('hats-') or config.get('surface_reconstruction'):
    # Existing painted wall texture is not a patch seam. Count only new
    # discontinuities above the source's local variation, with the same 8-RGB
    # tolerance; untouched rough wall must contribute zero seam evidence.
    source_error=rgb.astype(float)-replacement.astype(float)
    original_jump=np.maximum(np.linalg.norm(np.diff(source_error,axis=0,prepend=source_error[:1]),axis=2),np.linalg.norm(np.diff(source_error,axis=1,prepend=source_error[:,:1]),axis=2))
    jump=np.maximum(0,jump-original_jump)
   seam_fraction=float(np.mean(jump[samples]>8)) if samples.any() else 0
   metrics[iid]['seam_fraction']=round(seam_fraction,5)
   if seam_fraction>.02:why.append('patch seam check: abrupt wall transition')
   x,y,r,b=boxes[iid];rail_errors=[]
   rail_audit=cv2.dilate(m,np.ones((7,7),np.uint8))
   for row in ([] if rail_geometry.any() else np.where(rail_core)[0]):
    if row<y-24 or row>b+10:continue
    donor=rgb[row,outside[row]&(xx[row]>w*.15)&(xx[row]<w*.90)]
    if len(donor)>20:
     audit=hooks[row,x:r]==0
     if zid.startswith(('bags-','hats-')):
      # Clean retained rail grain is its own ground truth, not the median.
      # Audit only the reconstructed neighbourhood; lost-content below still
      # independently audits all non-wood pixels, including missed straps.
      audit &= rail_audit[row,x:r]>0
     rail_errors.extend((np.linalg.norm(base[row,x:r][audit].astype(float)-np.median(donor,axis=0),axis=1)>32).tolist())
   if rail_geometry.any():
    audit=rail_geometry&(rail_audit>0)&(hooks==0)
    rail_errors=(np.linalg.norm(base[audit].astype(float)-rail_paint[audit].astype(float),axis=1)>32).tolist()
   rail_background=1-float(np.mean(rail_errors)) if rail_errors else 1
   metrics[iid]['rail_background_fraction']=round(rail_background,4)
   if rail_background<.93:why.append('rail background check: silhouette fragments')
   # Hanging extraction is audited by the lost-content test below, including
   # its expanded rail-to-body region; a rail above a bbox is not itself failure.
   if why:failures[iid]=why
 # Lost-content audit complements the plate audit: a clean plate alone cannot
 # certify a cropped layer. Ignore antialias/shadow fringe, never whole rail rows.
 comp=base.astype(float).copy()
 silhouettes=np.zeros((h,w),np.uint8)
 for iid,m in masks.items():
  if reasons[iid]:continue
  comp[m>0]=rgb[m>0];silhouettes|=m
 feathered=base.astype(float).copy() if zid.startswith('bags-') else None
 if feathered is not None:
  for iid,m in masks.items():
   if reasons[iid]:continue
   alpha=cv2.GaussianBlur(m,(3,3),.55).astype(float)/255
   feathered=feathered*(1-alpha[:,:,None])+rgb*alpha[:,:,None]
 ratio=(rgb.astype(float)+1)/(base.astype(float)+1)
 shadow=(ratio.max(2)<.985)&(ratio.min(2)>.35)&(np.ptp(ratio,axis=2)<.10)
 lost=(np.linalg.norm(comp-rgb,axis=2)>30)&~shadow
 lost&=cv2.dilate(silhouettes,np.ones((5,5),np.uint8))==0
 # Wood grain changes under row reconstruction are expected; missing coloured
 # cloth over the rail is not. Exclude only source pixels matching wood donors.
 lost&=~wood
 for iid,(x,y,r,b) in boxes.items():
  if reasons[iid]:continue
  dx=round((r-x)*.15);dy=round((b-y)*.15)
  top=max(0,y-dy)
  if zid.startswith('bags-'):top=min(top,extraction_top[iid])
  region=lost[top:min(h,b+dy),max(0,x-dx):min(w,r+dx)]
  fraction=float(region.mean()) if region.size else 0
  metrics[iid]['lost_content_fraction']=round(fraction,4)
  if fraction>.01:failures.setdefault(iid,[]).append('rest check: lost content or wall reconstruction mismatch')
  if feathered is not None:
   actual=(np.linalg.norm(feathered-rgb,axis=2)>30)&~shadow
   region=actual[max(0,y-dy):min(h,b+dy),max(0,x-dx):min(w,r+dx)]
   feather_error=float(region.mean()) if region.size else 0
   metrics[iid]['feathered_rest_error']=round(feather_error,4)
   if feather_error>.01:failures.setdefault(iid,[]).append('rest check: feathered silhouette loses content')
 if failures:
  (REVIEW/f'{zid}-attempt-integrity.json').write_text(json.dumps(metrics,indent=2)+'\n')
  return reconstruct(rgb,boxes,zid,{**rejected,**failures})
 (REVIEW/f'{zid}-integrity.json').write_text(json.dumps(metrics,indent=2)+'\n')
 return base,masks,reasons

def review(zid,base,rgb,masks,reasons):
 im=Image.fromarray(base);im.thumbnail((1200,1200));im.save(REVIEW/f'{zid}-plate.jpg',quality=90)
 sheet=Image.new('RGB',(1000,240*((len(masks)+3)//4)), 'white');draw=ImageDraw.Draw(sheet)
 for j,(iid,m) in enumerate(masks.items()):
  y,x=np.where(m>0)
  if not len(x):continue
  box=(x.min(),y.min(),x.max()+1,y.max()+1);rgba=Image.fromarray(np.dstack([getattr(masks,'images',{}).get(iid,rgb),m])).crop(box);rgba.thumbnail((230,185))
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
