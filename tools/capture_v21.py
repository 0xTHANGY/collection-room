"""Requested deterministic desktop review captures; preview remains running."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'screenshots';BASE='http://127.0.0.1:8765/'
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--no-sandbox']);page=b.new_page(viewport={'width':1440,'height':900},device_scale_factor=1)
 page.goto(BASE,wait_until='networkidle');page.screenshot(path=str(OUT/'v2.1-hub.png'))
 catalog=json.loads((ROOT/'data/catalog.json').read_text())
 for zone,iid in [('hats-a','batch2-017'),('bags-a',None),('accessories-a',None)]+[(next(i['zone'] for i in catalog['items'] if i['id']==iid),iid) for iid in ['batch2-274','batch2-341']]:
  page.close();page=b.new_page(viewport={'width':1440,'height':900},device_scale_factor=1)
  z=next(z for z in catalog['zones'] if z['id']==zone);iid=iid or z['items'][0]
  page.goto(BASE+'?capture='+iid+'#/'+zone.replace('-','/'),wait_until='networkidle');page.wait_for_selector(f'[data-zone-id="{zone}"]');hot=page.locator(f'#zone-view .hotspot[data-item-id="{iid}"]');hot.scroll_into_view_if_needed();r=hot.bounding_box()
  page.mouse.move(r['x']-20,r['y']+r['height']/2);page.mouse.move(r['x']+r['width']+15,r['y']+r['height']/2,steps=5);hot.hover()
  if zone.startswith('accessories'):
   page.evaluate('document.getAnimations().forEach(a=>{a.pause();a.currentTime=50})')
  for f in [1,2]:
   if zone.startswith('accessories'):page.evaluate('(t)=>document.getAnimations().forEach(a=>{a.pause();a.currentTime=t})',f*70)
   else:page.wait_for_timeout(120)
   page.screenshot(path=str(OUT/f'v2.1-{zone}-{iid}-hover-{f}.png'))
  if iid in ['batch2-274','batch2-341']:
   hot.click();page.wait_for_function('()=>flipState==="open"');page.wait_for_function('()=>document.querySelector("#detail-image").naturalWidth>0');page.screenshot(path=str(OUT/f'v2.1-{iid}-card.png'));page.keyboard.press('Escape');page.wait_for_function('()=>!document.querySelector("dialog").open')
 page.goto(BASE+'#/hats/d',wait_until='networkidle');page.screenshot(path=str(OUT/'v2.1-hats-d.png'));b.close()
