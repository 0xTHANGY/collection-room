"""Reproducible part-B evidence, using metadata-free site assets."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'screenshots'
c=json.loads((ROOT/'data/catalog.json').read_text());items={i['id']:i for i in c['items']}
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',18)
def sheet(ids,height,size,name):
 out=Image.new('RGB',size,(235,235,235));draw=ImageDraw.Draw(out);x=10
 for iid in ids:
  im=Image.open(ROOT/items[iid]['image']).convert('RGBA');im=im.resize((int(im.width*height/im.height),height),Image.Resampling.LANCZOS)
  out.paste(im,(x,10),im);draw.text((x,height+16),iid,font=font,fill='black');x+=im.width+20
 out.save(OUT/name)
sheet(['batch2-'+n for n in ['039','102','234']],420,(731,460),'v2.1-after-dresses-lowres.png')
sheet(['cr-009'],700,(584,740),'v2.1-after-cr-009-cuffs.png')
sheet(['batch2-'+n for n in ['116','119','278','361','123','242']],360,(1570,400),'v2.1-after-accessories-hooks-tassels.png')
Image.fromarray(__import__('numpy').array(Image.open(ROOT/'assets/zones/shirts-a-1600.webp'))).save(OUT/'v2.1-after-shirts-a-wall.png')
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1440,'height':900},device_scale_factor=1,reduced_motion='reduce')
 page.goto('http://127.0.0.1:8765/#/shirts/a',wait_until='networkidle');page.screenshot(path=str(OUT/'v2.1-after-shirts-a.png'))
 page.locator('#zone-view .hotspot[data-item-id="cr-009"]').click();page.evaluate('document.getAnimations().forEach(a=>a.finish())');page.wait_for_function('()=>flipState==="open" && document.querySelector("#detail-image").naturalWidth>0');page.screenshot(path=str(OUT/'v2.1-after-card-cr-009.png'));b.close()
