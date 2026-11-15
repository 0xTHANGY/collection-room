"""Pixel-only, deterministic part-B recipes. Never reads modified card outputs."""
import cv2
import numpy as np
from PIL import Image, ImageFilter

def card(im, spec):
 a=np.array(im);rgb=a[:,:,:3].astype(float);alpha=a[:,:,3];h,w=alpha.shape
 method=spec['method']
 if method=='upscale':
  target=1200
  # Resize alpha independently; RGB is extended into transparent pixels first.
  rgb=cv2.inpaint(rgb.astype('uint8'),np.uint8(alpha<8)*255,3,cv2.INPAINT_TELEA)
  size=(round(w*target/max(h,w)),round(h*target/max(h,w)))
  color=Image.fromarray(rgb).resize(size,Image.Resampling.LANCZOS).filter(ImageFilter.UnsharpMask(radius=1.0,percent=55,threshold=3))
  al=Image.fromarray(alpha).resize(size,Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(.35))
  color.putalpha(al);return color
 if method=='cuff-clone':
  # Lettering occupies x=.920–.976, y=.932–.960. Clone clean parallel
  # stripes immediately to its left, preserving cuff perimeter and alpha.
  x0,x1=round(w*.912),round(w*.978);y0,y1=round(h*.938),round(h*.974)
  yy,xx=np.indices((h,w),dtype=np.float32)
  sx=w*.885+(xx-x0)*.32
  sy=yy+(xx-sx)*.20
  donor=cv2.remap(rgb.astype('uint8'),sx,sy,cv2.INTER_CUBIC).astype(float)
  mask=np.zeros((h,w),np.uint8);mask[y0:y1,x0:x1]=255
  blend=cv2.GaussianBlur(mask,(5,5),.8).astype(float)/255
  rgb=rgb*(1-blend[:,:,None])+donor*blend[:,:,None]
 if method=='hook':
  # Explicit above-garment hook limits, measured on source canvases.
  yend=round(h*spec['bottom']);x0=round(w*.35);x1=round(w*.65)
  region=np.zeros((h,w),bool);region[:yend,x0:x1]=True
  lum=rgb@np.array([.2126,.7152,.0722]);rgb[region]=np.repeat(lum[:,:,None],3,2)[region]
 if method=='fringe':
  # Alpha boundary band includes opaque edge hairs, never interior fabric.
  core=np.uint8(alpha>240);dist=cv2.distanceTransform(core,cv2.DIST_L2,3)
  edge=(alpha>0)&((alpha<240)|(dist<3))
  cast=np.maximum(0,np.minimum(rgb[:,:,0],rgb[:,:,2])-rgb[:,:,1]-3)
  strength=np.clip(cast/20,0,1)*edge
  # Remove the magenta component without changing green or alpha.
  for ch in [0,2]:rgb[:,:,ch]-=cast*strength
 return Image.fromarray(np.dstack([np.clip(rgb,0,255).astype('uint8'),alpha]))

def neutral_shirts(rgb, masks):
 f=rgb.astype(float);lo=f.min(2);hi=f.max(2)
 # Protect garments using existing extraction masks, and preserve wood/plant.
 protected=np.zeros(rgb.shape[:2],np.uint8)
 for mask in masks.values():protected=np.maximum(protected,mask)
 # Protect only the interior: warm wall contamination lives in the edge band.
 protected=cv2.erode(protected,np.ones((9,9),np.uint8)).astype(float)/255
 protected=cv2.GaussianBlur(protected,(5,5),1)
 warm=np.clip((45-(hi-lo))/12,0,1)*np.clip((lo-110)/35,0,1)
 weight=warm*(1-protected)
 # Keep original illumination/shadows; neutralise warm wall and floor.
 grey=f.mean(2)-7
 out=f*(1-weight[:,:,None])+grey[:,:,None]*weight[:,:,None]
 return np.clip(out,0,255).astype('uint8')
