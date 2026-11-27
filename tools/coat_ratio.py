import numpy as np
from PIL import Image
def spans(a):
    out=[]
    for r in a:
        xs=np.where(r)[0]; out.append(xs[-1]-xs[0]+1 if len(xs) else 0)
    return np.array(out)
def measure(path):
    a=np.asarray(Image.open(path).convert('RGBA'))[...,3]>60
    ys=np.where(a.any(1))[0]; top,bot=ys[0],ys[-1]; sp=spans(a)
    chest=np.median(sp[top+int(.35*(bot-top)):top+int(.45*(bot-top))+1])
    run=max(4,int(.025*(bot-top)))
    sh=next(y for y in range(top,bot) if (sp[y:y+run]>=.6*chest).all())
    hem=max(y for y in range(top,bot+1) if sp[y]>=.35*chest)
    L=hem+1-sh
    wc=float(np.median(sp[sh+int(.15*L):sh+int(.25*L)+1]))
    return dict(L=int(L),W=round(wc,1),ratio=round(L/wc,2),sh=int(sh),hem=int(hem),h=a.shape[0],w=a.shape[1],total=int(bot-top+1))
if __name__=='__main__':
    import sys
    for p in sys.argv[1:]: print(p.split('/')[-1],measure(p))
def sheet(paths,out,H=420):
    from PIL import ImageDraw
    tiles=[]
    for p in paths:
        m=measure(p); im=Image.open(p).convert('RGBA'); bg=Image.new('RGBA',im.size,(205,203,205,255)); bg.alpha_composite(im); im=bg.convert('RGB')
        d=ImageDraw.Draw(im); y1=m['sh']+int(.15*m['L']); y2=m['sh']+int(.25*m['L'])
        for y,c in ((m['sh'],'red'),(m['hem'],'red'),(y1,'blue'),(y2,'blue')): d.line([(0,y),(im.width,y)],fill=c,width=max(1,im.height//200))
        k=H/im.height; im=im.resize((round(im.width*k),H)); t=Image.new('RGB',(im.width,H+18),'white'); t.paste(im,(0,18)); ImageDraw.Draw(t).text((2,2),f"{p.split('/')[-1][:14]} {m['ratio']}",fill='black'); tiles.append(t)
    W=sum(t.width for t in tiles)+6*len(tiles); o=Image.new('RGB',(W,H+18),'white'); x=0
    for t in tiles: o.paste(t,(x,0)); x+=t.width+6
    o.save(out)
