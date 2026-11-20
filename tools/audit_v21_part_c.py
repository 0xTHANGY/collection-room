"""Audit final encoded plates for all zones, plus pixel-only metadata and review sheets."""
import csv,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]; REVIEW=ROOT.parent/'work/v2.1/review';FINAL=ROOT.parent/'final'
c=json.loads((ROOT/'data/catalog.json').read_text());g=json.loads((REVIEW/'gate.json').read_text())
finals={(r['category'],r['zone']):r for r in csv.DictReader((FINAL/'zone-finals.csv').open())}
report={'zones':{},'metadata_errors':[],'decode_errors':[]}
for z in c['zones']:
 plate=np.array(Image.open(ROOT/z['images'][0]['src']).convert('RGB'));h,w=plate.shape[:2]
 original=np.array(Image.open(FINAL/finals[z['category'],z['letter']]['path_in_zip']).convert('RGB').resize((w,h),Image.Resampling.LANCZOS))
 result={}
 for iid in z['items']:
  v=g[z['id']][iid]
  if not v['passed']:result[iid]={'passed':False,'reasons':v['reasons']};continue
  item=next(i for i in c['items'] if i['id']==iid);layer=item['layer'];alpha=np.array(Image.open(ROOT/layer['image']))[:,:,3]
  core=cv2.erode(np.uint8(alpha>180),np.ones((7,7),np.uint8))>0
  x=round(layer['left']*w/100);y=round(layer['top']*h/100);hh,ww=core.shape
  delta=np.linalg.norm(plate[y:y+hh,x:x+ww].astype(float)-original[y:y+hh,x:x+ww].astype(float),axis=2)
  changed=float(np.mean(delta[core]>24));assert changed>=.80,(iid,changed)
  prior=json.loads((REVIEW/f"{z['id']}-integrity.json").read_text())[iid]
  assert prior['background_fraction']>=.95,(iid,prior)
  assert prior.get('rail_background_fraction',1)>=.93,(iid,prior)
  assert prior.get('seam_fraction',0)<=.02,(iid,prior)
  result[iid]={'passed':True,'encoded_changed_fraction':round(changed,4),**prior}
 report['zones'][z['id']]=result
for p in (ROOT/'assets').rglob('*.webp'):
 try:
  with Image.open(p) as im:
   im.load()
   if any(k in im.info for k in ('exif','xmp','icc_profile')):report['metadata_errors'].append(str(p))
 except Exception as e:report['decode_errors'].append([str(p),str(e)])
report['webps_checked']=len(list((ROOT/'assets').rglob('*.webp')))
for cat in ('hats','bags','accessories'):
 zs=[z for z in c['zones'] if z['category']==cat];sheet=Image.new('RGB',(1200,290*((len(zs)+2)//3)), 'white');d=ImageDraw.Draw(sheet)
 for n,z in enumerate(zs):
  im=Image.open(REVIEW/f"{z['id']}-plate.jpg");im.thumbnail((390,260));x=n%3*400;y=n//3*290;sheet.paste(im,(x,y));d.text((x,y+263),z['id']+f" {sum(v['passed'] for v in g[z['id']].values())}/{len(g[z['id']])}",fill='black')
 sheet.save(REVIEW/f'{cat}-part-c-plates.jpg',quality=92)
(REVIEW/'part-c-integrity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'zones':len(report['zones']),'passing_items':sum(v['passed'] for z in report['zones'].values() for v in z.values()),'webps_checked':report['webps_checked'],'metadata_errors':report['metadata_errors'],'decode_errors':report['decode_errors']},indent=2))
