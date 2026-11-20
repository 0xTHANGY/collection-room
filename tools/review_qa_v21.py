"""Reproduce rejected hats-D code-only donor patch and sweater comparisons."""
import csv,json
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; FINAL=ROOT.parent/'final';OUT=ROOT.parent/'work/v2.1/review'
rows=list(csv.DictReader((FINAL/'zone-finals.csv').open()))
r=next(r for r in rows if r['category']=='hats' and r['zone']=='d');im=Image.open(FINAL/r['path_in_zip']).convert('RGB');a=np.array(im);h,w=a.shape[:2]
# Clone only existing wall/floor pixels; deliberately no invented geometry.
candidate=a.copy();right=int(w*.23);donor=a[:,int(w*.76):int(w*.76)+right];candidate[:,:right]=donor
compare=Image.new('RGB',(1200,450),'white')
for j,x in enumerate([im,Image.fromarray(candidate)]):
 x.thumbnail((600,420));compare.paste(x,(j*600,0))
d=ImageDraw.Draw(compare);d.text((0,425),'Original retained',fill='black');d.text((600,425),'REJECTED: donor repeats rail/items and breaks floor perspective',fill='black');compare.save(OUT/'hats-d-rejected.jpg',quality=90)
catalog=json.loads((ROOT/'data/catalog.json').read_text());out=Image.new('RGB',(1000,600),'#eeeeee');d=ImageDraw.Draw(out)
for j,iid in enumerate(['batch2-274','batch2-341']):
 item=next(i for i in catalog['items'] if i['id']==iid)
 for col,path in enumerate([FINAL/item['category']/'cutouts'/f'{iid}.png',ROOT/'assets/items/v2'/f'{iid}-600.webp']):
  x=Image.open(path).convert('RGBA');x.thumbnail((310,250));out.paste(x,(col*330,j*300),x);d.text((col*330,j*300+260),iid+(' before' if col==0 else ' after'),fill='black')
out.save(OUT/'sweaters-274-341.jpg',quality=90)
