"""Part C affected zone hover captures, including conservative fallbacks."""
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1440,'height':900},device_scale_factor=1)
 for zone,ids in [('bags-e',['batch2-384','batch2-385','batch2-386','batch2-387']),('bags-b',['batch2-362']),('shirts-a',['batch2-328','batch2-332','batch2-333','cr-009'])]:
  for iid in ids:
   page.goto('http://127.0.0.1:8765/#/'+zone.replace('-','/'),wait_until='networkidle')
   hot=page.locator(f'#zone-view .hotspot[data-item-id="{iid}"]');hot.hover()
   page.wait_for_timeout(250)
   page.screenshot(path=str(ROOT/'screenshots'/f'v2.1-{zone}-{iid}-hover-1.png'))
 b.close()
