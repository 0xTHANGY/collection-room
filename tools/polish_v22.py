"""Deterministic local v2.2 visor and chroma-key asset polish; writes site only."""
from pathlib import Path
import sys,json
import cv2
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'data/source'))
from overrides_v21 import card, visor_slits

def save(im,path):
 Image.fromarray(np.array(im)).save(path,'WEBP',lossless=True,method=6)

def run():
 out=ROOT/'screenshots';out.mkdir(exist_ok=True)
 im=Image.open(ROOT.parent/'final/hats/cutouts/batch2-192.png').convert('RGBA')
 fixed=card(im,{'method':'visor-slits'})
 for label,hat in [('before',im),('after',fixed)]:
  bg=Image.new('RGBA',hat.size,(225,225,225,255));bg.alpha_composite(hat);bg.convert('RGB').save(out/f'v2.2-{label}-192-card.png')
 fixed.crop((350,500,530,595)).resize((1080,570)).save(ROOT/'work/v2.2/after-card-detail.png')
 zone=Image.open(ROOT/'data/source/generated/hats-zone-d-v22.webp').convert('RGB')
 fixedzone=visor_slits(zone,'zone')
 save(fixedzone,ROOT/'data/source/generated/hats-zone-d-v22-192fix.webp')
 for label,img in [('before',zone),('after',fixedzone)]:
  img.crop((1222,167,1445,398)).resize((669,693),Image.Resampling.LANCZOS).save(out/f'v2.2-{label}-192-zone.png')
 fixedzone.crop((1300,235,1360,265)).resize((960,480)).save(ROOT/'work/v2.2/after-zone-detail.png')
 manifest=ROOT/'data/source/generated-v22.json';data=json.loads(manifest.read_text());data['zones']['hats-d']='data/source/generated/hats-zone-d-v22-192fix.webp';manifest.write_text(json.dumps(data,indent=2)+'\n')
 for n in ['039','102','234']:
  rgb=np.array(Image.open(ROOT/f'work/v2.2/gen/dress-{n}-v22.png').convert('RGB')).astype(np.float32)
  # Pure magenta compositing: G contains only foreground; excess R/B
  # measures the background contribution. Preserve warm and colored fabric.
  cast=np.maximum(0,np.minimum(rgb[:,:,0],rgb[:,:,2])-rgb[:,:,1])
  alpha=1-np.clip(cast/255,0,1)
  # Discard near-pure key background while retaining fine translucent edges.
  alpha[cast>195]=0
  interior=cv2.distanceTransform(np.uint8(alpha>.5),cv2.DIST_L2,3)>3
  alpha[interior]=1
  clean=(rgb-(1-alpha[:,:,None])*np.array([255,0,255]))/np.maximum(alpha[:,:,None],.001)
  clean=np.clip(clean,0,255)
  # Neutral dark tulle: unmix against the dark sleeve rather than magenta.
  if n=='039':
   yy,xx=np.indices(alpha.shape)
   cuffs=(yy>430)&(yy<585)&(((xx>140)&(xx<400))|((xx>624)&(xx<880)))
   sheer=cuffs&(cast>2)&(alpha>0)
   # A neutral black tulle at ~45 luminance explains its partial coverage.
   coverage=np.clip((255-cast)/(255-42),0,1)
   alpha[sheer]=coverage[sheer]
   grey=np.clip(rgb[:,:,1]/np.maximum(coverage,.001),15,75)
   clean[sheer]=np.repeat(grey[:,:,None],3,axis=2)[sheer]
   neutral=cuffs&(alpha>0)
   clean[neutral]=np.repeat(clean[:,:,1,None],3,axis=2)[neutral]
  rgba=np.dstack([clean.astype(np.uint8),np.round(alpha*255).astype(np.uint8)])
  save(Image.fromarray(rgba),ROOT/f'data/source/generated/dress-{n}-v22.webp')
  bg=Image.new('RGBA',(1024,1536),(220,220,220,255));bg.alpha_composite(Image.fromarray(rgba));bg.convert('RGB').resize((512,768)).save(ROOT/f'work/v2.2/dress-{n}-after.png')
if __name__=='__main__':run()
