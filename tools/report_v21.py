"""Summarize gate decisions without changing assets."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REVIEW=ROOT.parent/'work/v2.1/review'
g=json.loads((REVIEW/'gate.json').read_text())
def compact(ids):
 nums=sorted(int(i[7:]) for i in ids if i.startswith('batch2-'));chunks=[]
 for n in nums:
  if chunks and n==chunks[-1][-1]+1:chunks[-1].append(n)
  else:chunks.append([n])
 return ', '.join([f'{c[0]:03d}'+(f'–{c[-1]:03d}' if len(c)>1 else '') for c in chunks]+[i for i in ids if not i.startswith('batch2-')])
reasons={'rail geometry reconstruction unvalidated':'G','patch seam check: abrupt wall transition':'E','rail background check: silhouette fragments':'L','bench boundary: wood reconstruction unvalidated':'T','plate removal check: unchanged foreground':'R','plate background check: residual foreground':'P','upper strap outside extraction: incomplete silhouette':'S','wall reconstruction not validated for this background':'W','floor or furnishing reconstruction unsupported':'F','mask coverage out of range':'M','A-frame background requires redrawing':'A','item touches image boundary':'B','touches neighbour mask':'N','missing pale handle or upper silhouette':'H'}
lines=['Batch2 IDs omit `batch2-`. W = background not validated; F = floor/furnishing; M = mask coverage; A = A-frame; B = image boundary; N = mask collision; H = missing handle/upper silhouette; T = bench boundary unvalidated; R = insufficient removal; P = residual foreground; S = incomplete upper strap; L = rail fragments; E = patch seam; G = rail geometry.','', '| Zone | Pass/total | Failing IDs (reason) |','|---|---:|---|']
for z,items in g.items():
 groups={}
 for iid,v in items.items():
  if not v['passed']:groups.setdefault('+'.join(reasons[r] for r in v['reasons']),[]).append(iid)
 lines.append(f"| {z} | {sum(v['passed'] for v in items.values())}/{len(items)} | "+'; '.join(compact(ids)+' ('+r+')' for r,ids in groups.items())+' |')
(REVIEW/'gate-compact.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
