#!/usr/bin/env python3
"""Verify the local v2 site; preview must already serve site/ on port 8765."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
BASE='http://127.0.0.1:8765/'
def main():
    catalog=json.loads((ROOT/'data/catalog.json').read_text())
    out=ROOT/'screenshots';out.mkdir(exist_ok=True)
    report={'zones':[], 'items_checked':0,'hotspots_checked':0,'list_only_checked':0,'rest_layers_checked':0,'errors':[],'console_errors':[],'page_errors':[],'failed_requests':[],'http_errors':[]}
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':900},device_scale_factor=1,reduced_motion='reduce')
        page.on('console',lambda m:report['console_errors'].append(m.text) if m.type=='error' else None)
        page.on('pageerror',lambda e:report['page_errors'].append(str(e)))
        page.on('requestfailed',lambda r:report['failed_requests'].append({'url':r.url,'failure':r.failure}))
        page.on('response',lambda r:report['http_errors'].append({'url':r.url,'status':r.status}) if r.status>=400 else None)
        responses=set()
        page.on('response',lambda r:responses.add(r.url) if r.ok else None)
        page.goto(BASE,wait_until='networkidle');page.wait_for_selector('#room-tiles a');page.screenshot(path=str(out/'v2.1-hub.png'))
        paths=[u.removeprefix(BASE).split('?')[0] or 'index.html' for u in responses if u.startswith(BASE)]
        first_load=sum((ROOT/f).stat().st_size for f in paths if (ROOT/f).is_file())
        report['first_load_bytes']=first_load;report['first_load_files']=sorted(paths)
        report['lazy_initial_cards']=not any(f.startswith('assets/items/') for f in paths)
        report['lazy_initial_zone_layers']=not any(f.startswith('assets/zones/layers/') for f in paths)
        extra_zone=next((z['id'] for z in catalog['zones'] if z['has_layout'] and z['id'] not in ('hats-a','jackets-a')),None)
        for zone in catalog['zones']:
            print('Verifying', zone['id'], flush=True)
            page.goto(BASE+f"#/{zone['category']}/{zone['letter']}",wait_until='networkidle')
            page.wait_for_selector(f'[data-zone-id="{zone["id"]}"]')
            passing=page.locator('#zone-view .layer-wrap:not(.layer-fallback)')
            report['rest_layers_checked']+=passing.count()
            invisible=passing.evaluate_all('els=>els.filter(e=>Number(getComputedStyle(e).opacity)<.99).map(e=>e.dataset.layerItemId)')
            if invisible:report['errors'].append(f"invisible passing layers: {zone['id']} {invisible}")
            pairs=page.locator('#zone-view .hotspot').evaluate_all('''els => {const r=els.map(e=>({id:e.dataset.itemId,r:e.getBoundingClientRect()})); const pairs=[]; for(let i=0;i<r.length;i++)for(let j=i+1;j<r.length;j++){const a=r[i].r,b=r[j].r;if(Math.min(a.right,b.right)-Math.max(a.left,b.left)>.1&&Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>.1)pairs.push([r[i].id,r[j].id]);}return pairs;}''')
            report['zones'].append({'id':zone['id'],'overlap_pairs':pairs,'has_layout':zone['has_layout'],'hotspots':page.locator('#zone-view .hotspot').count()})
            if pairs:report['errors'].append(f"overlaps: {zone['id']} {pairs}")
            page.evaluate('window.scrollTo(0,0)')
            if zone['id'] in ('hats-a','bags-a','accessories-a','hats-d',extra_zone):
                page.screenshot(path=str(out/f'v2.1-zone-{zone["id"]}.png'))
            for iid in zone['items']:
                item=next(i for i in catalog['items'] if i['id']==iid)
                selector=f'#zone-view .hotspot[data-item-id="{iid}"]' if item['hotspot'] else f'#zone-view .list-item[data-item-id="{iid}"]'
                trigger=page.locator(selector)
                try:
                    trigger.click()
                    page.wait_for_function('(id)=>document.querySelector("dialog").open && document.querySelector("dialog").dataset.itemId===id',arg=iid)
                    assert page.locator('#detail-name').inner_text()==item['name']
                    assert page.locator('#detail-story').inner_text()=='故事待补充'
                    assert page.locator('#detail-brand').inner_text()==item['brand']
                    page.evaluate('document.getAnimations().forEach(a=>a.finish())')
                    page.wait_for_function('() => flipState === "open"')
                    page.wait_for_function('() => document.querySelector("#detail-image").naturalWidth > 0 && !document.querySelector("#detail-image").hidden')
                    if iid==catalog['zones'][0]['items'][0]:page.screenshot(path=str(out/'v2.1-card-open.png'))
                    page.locator('#close-dialog').click();page.evaluate('document.getAnimations().forEach(a=>a.finish())');page.wait_for_function('() => !document.querySelector("dialog").open')
                    page.wait_for_function('() => !location.hash.startsWith("#/item/")')
                    assert trigger.evaluate('(e)=>document.activeElement===e')
                    report['items_checked']+=1
                    report['hotspots_checked' if item['hotspot'] else 'list_only_checked']+=1
                except Exception as e:
                    report['errors'].append(f'{zone["id"]}/{iid}: {e}')
                    page.evaluate('document.querySelector("dialog").close()')
        # Exercise the original ten hub hotspots after visiting every category,
        # including IDs shared by hub/catalog layers.
        page.goto(BASE+'#/',wait_until='networkidle')
        hub=json.loads((ROOT/'data/items.json').read_text());report['hub_items_checked']=0
        for item in hub:
            trigger=page.locator(f'#hotspots button[data-item-id="{item["id"]}"]')
            trigger.click();page.wait_for_function('() => document.querySelector("dialog").open')
            assert page.locator('#item-dialog').get_attribute('data-item-id')==item['id']
            assert page.locator('#detail-name').inner_text()==item['display_name']
            page.evaluate('document.getAnimations().forEach(a=>a.finish())');page.wait_for_function('() => flipState === "open"')
            page.keyboard.press('Escape');page.evaluate('document.getAnimations().forEach(a=>a.finish())');page.wait_for_function('() => !document.querySelector("dialog").open')
            assert trigger.evaluate('(e)=>document.activeElement===e')
            report['hub_items_checked']+=1
        # Run one full-duration ordinary-motion FLIP in addition to accelerated fades.
        page.emulate_media(reduced_motion='no-preference')
        page.locator('#hotspots button').first.click();page.wait_for_function('() => flipState === "open"')
        page.keyboard.press('Escape');page.wait_for_function('() => !document.querySelector("dialog").open')
        report['normal_motion_flip']=True
        page.emulate_media(reduced_motion='reduce')
        # Deep links and browser history; keyboard activation and Escape focus return.
        first=next(i for i in catalog['items'] if i['hotspot'])
        page.goto(BASE+'#/item/'+first['id'],wait_until='networkidle');page.wait_for_function('() => document.querySelector("dialog").open');page.evaluate('document.getAnimations().forEach(a=>a.finish())');page.wait_for_function('() => flipState === "open"');page.keyboard.press('Escape');page.evaluate('document.getAnimations().forEach(a=>a.finish())');page.wait_for_function('() => !document.querySelector("dialog").open')
        page.goto(BASE+'#/');page.wait_for_selector('#room-tiles a');page.locator('#room-tiles a').first.click();page.wait_for_selector('#zone-view:not([hidden])');page.go_back();page.wait_for_selector('#hub-view:not([hidden])');page.go_forward();page.wait_for_selector('#zone-view:not([hidden])')
        page.locator('#zone-view .hotspot').first.focus();page.keyboard.press('Enter');page.wait_for_function('() => document.querySelector("dialog").open');page.evaluate('document.getAnimations().forEach(a=>a.finish())');page.wait_for_function('() => flipState === "open"');page.keyboard.press('Escape');page.evaluate('document.getAnimations().forEach(a=>a.finish())');page.wait_for_function('() => !document.querySelector("dialog").open')
        page.set_viewport_size({'width':375,'height':812});page.goto(BASE+'#/hats/a',wait_until='networkidle');report['mobile_horizontal_overflow']=page.evaluate('document.documentElement.scrollWidth > innerWidth')
        report['extra_layout_screenshot']='unavailable: only hats-a and jackets-a have supplied layouts' if sum(z['has_layout'] for z in catalog['zones'])==2 else 'saved'
        browser.close()
    report['passed']=not any(report[k] for k in ('errors','console_errors','page_errors','failed_requests','http_errors')) and report['lazy_initial_cards'] and report['lazy_initial_zone_layers'] and not report['mobile_horizontal_overflow'] and report['items_checked']==len(catalog['items'])
    (out/'v2.1-verify.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    summary_path=ROOT/'data/build-v2-summary.json';summary=json.loads(summary_path.read_text());summary['first_load_bytes']=first_load;summary['first_load_files']=report['first_load_files'];summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('zones','first_load_files')},ensure_ascii=False,indent=2))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
