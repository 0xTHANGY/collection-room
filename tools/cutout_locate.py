"""Find the original painting (git commit + zone image + pixel box) a layer was cut from (P3, 2026-11-24)."""
import json,subprocess,io,sys,numpy as np
from PIL import Image
def git(*a): return subprocess.run(['git',*a],capture_output=True).stdout
def show(c,p):
    r=subprocess.run(['git','show',f'{c}:{p}'],capture_output=True); return r.stdout if r.returncode==0 else None
cats={}
def cat(c):
    if c not in cats: cats[c]=json.loads(show(c,'data/catalog.json'))
    return cats[c]
def locate(iid):
    lc=git('log','-1','--format=%h','--','assets/zones/layers/'+iid+'.webp').decode().strip()
    lay=np.array(Image.open(f'assets/zones/layers/{iid}.webp').convert('RGBA')).astype(int)
    c=cat(lc); it=[i for i in c['items'] if i['id']==iid][0]; L=it['layer']; zid=it['zone']
    z=[z for z in c['zones'] if z['id']==zid][0]; src=max(z['images'],key=lambda x:x['width'])['src']
    best=None
    for zc in git('log','--format=%h',lc,'--',src).decode().split():
        b=show(zc,src)
        im=np.array(Image.open(io.BytesIO(b)).convert('RGB')).astype(int); H,W=im.shape[:2]
        x=round(L['left']/100*W);y=round(L['top']/100*H);w=round(L['width']/100*W);h=round(L['height']/100*H)
        if (w,h)!=(lay.shape[1],lay.shape[0]): 
            continue
        cr=im[y:y+h,x:x+w]; m=lay[:,:,3]>200
        if cr.shape[:2]!=m.shape: continue
        d=np.abs(cr-lay[:,:,:3])[m].mean()
        if best is None or d<best[0]: best=(round(d,2),zc,src,(x,y,w,h),zid,lc)
    return best
if __name__=='__main__':
    for iid in sys.argv[1:]: print(iid,locate(iid))
