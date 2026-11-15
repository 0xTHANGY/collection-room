"""Garment extraction on free-standing racks, separate from wall accessories.

Geometry is in 1600px painting coordinates. Masks retain single ownership of
visible pixels; optional RGB layers complete cloth hidden by another garment.
The opt-in deep_overlap path handles sleeves mostly hidden by a front garment;
ordinary rack configs retain their original extraction and completion.
"""
import cv2
import numpy as np


class GarmentMasks(dict):
    """The normal mask dictionary, with optional completed RGB per item."""
    def __init__(self):
        super().__init__()
        self.images = {}


def attached_background_fraction(pixels, plate, mask):
    """Find smooth wall patches attached to a cutout, not cream cloth.

    Fabric can have the exact mean colour of plaster. Require a smooth patch
    (at painting resolution), colour agreement, and attachment to the edge.
    This retains the wall-wedge check without treating printed/folded cream
    panels in the middle of a garment as unchanged background.
    """
    value=pixels.astype(np.float32)
    mean=cv2.blur(value,(7,7))
    variation=np.sqrt(np.maximum(0,cv2.blur(value*value,(7,7))-mean*mean).mean(2))
    wall=(np.linalg.norm(value-plate.astype(float),axis=2)<14)&(variation<1.6)&(mask>0)
    n,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(wall))
    edge=(mask>0)&~cv2.erode(np.uint8(mask>0),np.ones((3,3),np.uint8)).astype(bool)
    touching=set(np.unique(labels[edge&wall]))-{0}
    # Isolated matching pixels along an antialiased edge aren't wall wedges.
    touching={k for k in touching if stats[k,4]>=16}
    return float(np.isin(labels,list(touching)).sum()/max(1,np.count_nonzero(mask)))


def wall_model(rgb, config, occupied):
    h, w = rgb.shape[:2]
    yy, xx = np.indices((h, w))
    samples = np.zeros((h, w), np.uint8)
    for x, y, r, b in config['wall_donors']:
        samples[y:b, x:r] = 1
    samples &= ~occupied
    # Fit broad painted illumination, rejecting sparse shadows and wood grain.
    x, y = xx / w, yy / h
    features = [np.ones_like(x), x, y, x*x, x*y, y*y]
    sel = samples > 0
    A = np.column_stack([f[sel] for f in features])
    values = rgb[sel].astype(float)
    keep = np.ones(len(values), bool)
    for _ in range(4):
        coeff = np.linalg.lstsq(A[keep], values[keep], rcond=None)[0]
        err = np.linalg.norm(A @ coeff - values, axis=1)
        keep = err < max(8, np.quantile(err, .85))
    return np.uint8(np.clip(sum(f[:, :, None]*c for f,c in zip(features,coeff)),0,255))


def reconstruct(rgb, boxes, config):
    h,w = rgb.shape[:2]
    yy,xx = np.indices((h,w))
    occupied=np.zeros((h,w),np.uint8)
    for x,y,r,b in boxes.values():
        occupied[max(0,y-3):min(h,b+15),max(0,x-4):min(w,r+4)]=1
    wall=wall_model(rgb,config,occupied)
    background=wall.copy()
    post_region=np.zeros((h,w),bool)
    for x0,y0,x1,y1,width,dy0,dy1 in config.get('posts',[]):
        center=x0+(yy-y0)*(x1-x0)/(y1-y0)
        signed=xx-center
        offset=np.rint(signed).astype(int)
        profile=[]
        for dx in range(-width-5,width+6):
            rows=np.arange(dy0,dy1)
            cols=np.rint(x0+(rows-y0)*(x1-x0)/(y1-y0)+dx).astype(int)
            profile.append(np.median(rgb[rows,cols],axis=0))
        profile=np.uint8(profile)
        reg=(np.abs(offset)<=width+5)&(yy>=y0)&(yy<y1)
        pos=np.clip(signed[reg]+width+5,0,len(profile)-1)
        low=np.floor(pos).astype(int); high=np.minimum(low+1,len(profile)-1)
        t=(pos-low)[:,None]
        background[reg]=profile[low]*(1-t)+profile[high]*t
        post_region|=reg
    structure=np.zeros((h,w),np.uint8)
    for x,y,r,b in config['structure']:
        structure[y:b,x:r]=1
    # Cloth differs from the wall in colour or luminance. Smooth, neutral
    # cast shadows are excluded; textured pale fabric remains foreground.
    delta=np.linalg.norm(rgb.astype(float)-background.astype(float),axis=2)
    ratio=(rgb.astype(float)+1)/(background.astype(float)+1)
    shadows=(np.ptp(ratio,axis=2)<.045)&(ratio.mean(2)>.70)&(ratio.mean(2)<1)
    shadows[post_region]=False
    # Exposed post grain is warmer than textile shadows. A wall-difference
    # mask alone sees that grain as a narrow sleeve extension beside a post.
    red,green,blue=rgb.astype(int).transpose(2,0,1)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV).astype(float) if config.get('deep_overlap') else None
    wood=post_region&(red-green>28)&(green-blue>18)&(green-blue<36)
    wood=cv2.dilate(np.uint8(wood),np.ones((3,3),np.uint8)).astype(bool)&post_region
    fg=(delta>19)&~shadows&~structure.astype(bool)
    masks=GarmentMasks(); reasons={}
    costs=[]; candidates=[]
    hooks=np.zeros((h,w),bool)
    for iid,(x,y,r,b) in boxes.items():
        # Two stacked rows can share the same horizontal centre. Select the
        # rail above the garment by vertical distance, not centre-x alone.
        rail=min(config['rails'],key=lambda z: abs(y-z[3])*4 + (0 if z[0]< (x+r)/2 < z[2] else w))
        top=rail[3]+3
        # Metal hooks stay with the rack; the entire wooden hanger moves.
        # Find the first broad warm wood row, not a bbox-dependent stem cut.
        warm=(red-blue>48)&(red-green>24)&(rgb.mean(2)<200)&(np.abs(xx-(x+r)/2)<20)
        warm&=(yy>=rail[3])&(yy<y+8)
        woodrows=np.flatnonzero(warm.sum(1)>=3)
        hanger_top=int(woodrows[0]) if len(woodrows) else top+6
        apex=np.flatnonzero(warm[hanger_top])
        hook_x=float(np.median(apex)) if len(apex) else (x+r)/2
        hooks|=(np.abs(xx-hook_x)<=3)&(yy>=rail[3]-1)&(yy<hanger_top)
        roi=(xx>=x-8)&(xx<r+8)&(yy>=top)&(yy<b+10)
        # Local GrabCut separates pale cuffs from wall and retains fine hanger
        # edges without turning the triangular hanger opening into opaque wall.
        seeds=np.zeros((h,w),np.uint8)
        seeds[roi]=cv2.GC_PR_BGD
        seeds[roi&fg]=cv2.GC_PR_FGD
        core=(xx>x+(r-x)*.25)&(xx<r-(r-x)*.25)&(yy>y+(b-y)*.15)&(yy<b-(b-y)*.12)&fg
        seeds[core]=cv2.GC_FGD
        seeds[structure>0]=cv2.GC_BGD
        # A crop is essential: a full-size GMM can learn neighbouring clothes
        # as wall, even though they are outside this item's allowed silhouette.
        px,py=max(0,x-12),max(0,top-4);pr,pb=min(w,r+12),min(h,b+14)
        s=seeds[py:pb,px:pr].copy()
        cv2.setRNGSeed(23)  # stable across build category/zone iteration order
        cv2.grabCut(rgb[py:pb,px:pr].copy(),s,None,np.zeros((1,65)),np.zeros((1,65)),4,cv2.GC_INIT_WITH_MASK)
        m=np.zeros((h,w),np.uint8);m[py:pb,px:pr]=np.uint8((s==1)|(s==3))
        m[~roi]=0;m[structure>0]=0
        # Wall openings remain transparent even if enclosed by hanger wood.
        if config.get('pale_cloth'):
            # Cream sleeves share the wall colour. GrabCut uses their folds
            # and borders; a global wall/shadow colour veto amputates them.
            m[((delta<15)|shadows)&(yy<y+8)]=0
        else:
            m[(delta<15)|shadows]=0
        n,lab,stats,_=cv2.connectedComponentsWithStats(m)
        for k in range(1,n):
            if stats[k,4]<8:m[lab==k]=0
        m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        m=cv2.dilate(m,np.ones((3,3),np.uint8))
        m[~roi|(structure>0)|(yy<hanger_top)]=0
        # Ownership in overlapping boxes follows material sampled from the
        # unoccluded centre; the spatial term only breaks similar-colour ties.
        colors=np.float32(rgb[core])
        cv2.setRNGSeed(23)
        _,_,centers=cv2.kmeans(colors, min(5,len(colors)),None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,.5),3,cv2.KMEANS_PP_CENTERS)
        colorcost=np.min([np.linalg.norm(rgb.astype(float)-c,axis=2) for c in centers],axis=0)
        upper=(m>0)&(yy<y+9)&(yy>top)
        axis=float(np.median(xx[upper])) if upper.any() else (x+r)/2
        mirror_x=np.clip(np.rint(2*axis-xx).astype(int),0,w-1)
        symmetry=np.linalg.norm(rgb.astype(float)-rgb[yy,mirror_x].astype(float),axis=2)
        weight=np.where((b-y>2*(r-x))&(yy>y+(b-y)*.5),.7,.35)
        cost=colorcost*(1-weight)+symmetry*weight+12*np.abs(xx-(x+r)/2)/max(1,r-x)
        # Reject exposed wood only when the opposite sleeve confirms a
        # different material; warm pink/ochre cloth can share wood's hue.
        exposed=wood&(symmetry>45)&(colorcost>30)
        if config.get('deep_overlap'):
            # Patterned collars can share the post's warm hue. Require
            # agreement with the donor wood and stay below the collar.
            exposed &= (np.linalg.norm(rgb.astype(float)-background.astype(float),axis=2)<35)&(yy>y+(b-y)*.2)
        m[exposed]=0
        candidates.append(m>0)
        if config.get('complete_overlap'):
            # Printed seams and quilting must not make a serrated ownership
            # boundary. Average material evidence before comparing neighbours.
            cost=cv2.GaussianBlur(cost,(7,7),1.4)
        if config.get('deep_overlap'):
            sample=hsv[core]
            hue=float(np.median(sample[:,0]));sat=float(np.median(sample[:,1]))
            hue_delta=np.minimum(np.abs(hsv[:,:,0]-hue),180-np.abs(hsv[:,:,0]-hue))
            if sat>140 and np.mean(np.abs(sample[:,0]-hue)<12)>.85:
                # A bright foreground fold must not remain as a cyan/pink
                # seam on a dark rear coat when the front garment swings.
                material=(hue_delta<12)&(hsv[:,:,1]>sat*.65)
                cost[material]-=60
                # Checked neighbour cloth and exposed post wood are not
                # part of a uniformly saturated sleeve, even at its cuff.
                alien=(hue_delta>20)&(yy>y+(b-y)*.25)&(hsv[:,:,1]<sat*.7)&(np.abs(xx-axis)>(r-x)*.25)
                m[alien]=0
                candidates[-1]=m>0
        cost[~(m>0)]=1e6;costs.append(cost)
    allcost=np.stack(costs);owner=allcost.argmin(0);union=np.stack(candidates).any(0)
    # A dark edge on the neighbouring coat can match this garment's palette.
    # Reattach thin row fragments to the adjacent broad silhouette; do not
    # let a matching black cuff/hem become a long loose strip on another coat.
    for k,(iid,(x,y,r,b)) in enumerate(boxes.items()):
        for row in range(max(0,round(y+(b-y)*(.25 if b-y>2*(r-x) else .5))),min(h,b+10)):
            own=union[row]&(owner[row]==k)
            edges=np.diff(np.r_[False,own,False].astype(int))
            runs=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
            if not runs or max(e-a for a,e in runs)<(r-x)*.4:continue
            for a,e in runs:
                if e-a>(r-x)*.12:continue
                near=np.r_[max(0,a-3):a,e:min(w,e+3)]
                others=[j for j in np.unique(owner[row,near][union[row,near]]) if j!=k]
                for j in others:
                    ox,oy,orr,ob=list(boxes.values())[j]
                    if config.get('deep_overlap') and not candidates[j][row,a:e].all():continue
                    if ox-8<=a and e<=orr+8 and oy<=row<ob+10:
                        owner[row,a:e]=j
                        break
    for k,(iid,box) in enumerate(boxes.items()):
        m=np.uint8(union&(owner==k))*255
        # Detached pieces in an adjacent bbox are not part of this garment.
        # The hanger and real drawcords remain attached to the main silhouette.
        n,labels,stats,_=cv2.connectedComponentsWithStats(m)
        if n>1:
            main=1+np.argmax(stats[1:,4])
            for component in range(1,n):
                a,c,d,e,area=stats[component]
                hanger=c<box[1]+8 and abs(a+d/2-(box[0]+box[2])/2)<(box[2]-box[0])*.3
                if component!=main and not hanger:m[labels==component]=0
        # Preserve small enclosed cloth print/pixel holes, but never hanger gaps.
        x,y,r,b=box
        inverse=np.uint8(m==0);n,lab,stats,_=cv2.connectedComponentsWithStats(inverse)
        for j in range(1,n):
            a,c,d,e,area=stats[j]
            if area<(r-x)*(b-y)*.15 and c>y+8 and a>x and a+d<r and c+e<b:
                free=(lab==j)&~union;m[free]=255
        # Thin post-grain remnants are not sleeve details. Regularise only
        # the known post band, leaving drawcords and garment hems elsewhere.
        opened=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((7,7),np.uint8))
        if not config.get('deep_overlap'):
            m[post_region]=opened[post_region]
        masks[iid]=m
        reasons[iid]=[]
        area=np.count_nonzero(m)/max(1,(r-x)*(b-y))
        if not .25<area<1.15:reasons[iid].append('garment silhouette coverage out of range')
    if config.get('deep_overlap'):
        # Ownership cleanup can discard a tiny neighbour-coloured island
        # enclosed by real cloth. Return that now-unowned source pixel to
        # its enclosing garment; never fill over another visible owner.
        owned=np.logical_or.reduce([m>0 for m in masks.values()])
        for iid,m in masks.items():
            x,y,r,b=boxes[iid]
            n,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(m==0))
            for j in range(1,n):
                a,c,d,e,area=stats[j]
                if area<max(32,(r-x)*(b-y)*.003) and c>y+8 and a>x and a+d<r and c+e<b:
                    fill=(labels==j)&~owned
                    m[fill]=255;owned[fill]=True
    masks.order=config.get('depth_order',list(masks))
    visible={iid:m.copy() for iid,m in masks.items()}
    for depth,iid in enumerate(masks.order):
        x,y,r,b=boxes[iid]
        m=visible[iid]
        # The hanger apex is a better symmetry axis than an overlapping bbox.
        top_rows=(yy<y+9)&(yy>config['rails'][0][3]+8)&(m>0)
        center=float(np.median(xx[top_rows])) if top_rows.any() else (x+r)/2
        reflected_x=np.clip(np.rint(2*center-xx).astype(int),0,w-1)
        reflected=m[yy,reflected_x]>0
        front=np.zeros((h,w),bool)
        for other in masks.order[depth+1:]:front|=visible[other]>0
        envelope=np.zeros((h,w),bool)
        for row in range(max(0,y+12),min(h,b+3)):
            run=np.flatnonzero(m[row])
            if len(run):envelope[row,run[0]:run[-1]+1]=True
        substantial=config.get('deep_overlap') and ((reflected|envelope)&front).sum(1).max()>12
        if substantial:
            # Only expand a genuine occlusion, never bridge a wall gap to
            # an adjacent garment. This closes ownership/antialiasing slots.
            front=cv2.dilate(np.uint8(front),np.ones((17,17),np.uint8)).astype(bool)
        hidden=(reflected|envelope)&front&(yy>y+12)&(yy<b+3)&(xx>x-8)&(xx<r+8)
        if config.get('deep_overlap'):
            hidden &= m==0
        # Symmetry alone can reach across a real wall gap onto the next
        # coat. Only complete a row where the occluder touches this garment;
        # otherwise the invented strip floats when the neighbour moves.
        for row in np.flatnonzero(hidden.any(1)):
            joined=(m[row]>0)|hidden[row]
            edge=np.diff(np.r_[False,joined,False].astype(int))
            for a,e in zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)):
                if not substantial and not m[row,a:e].any():hidden[row,a:e]=False
        # A real wall gap can terminate an occlusion at the curved hem.
        # Taper only the invented extension toward that gap, rather than
        # ending it in a square tab when the front coat's hem stops touching.
        active=np.uint8(hidden.any(1))[:,None]
        vertical=cv2.distanceTransform(active,cv2.DIST_L2,3)[:,0]
        for row in np.flatnonzero(hidden.any(1)):
            if substantial or row<=y+(b-y)*.5 or vertical[row]>=10:continue
            own=np.flatnonzero(m[row])
            extra=np.flatnonzero(hidden[row])
            if not len(own) or not len(extra):continue
            distance=np.min(np.abs(extra[:,None]-own[None,:]),axis=1)
            limit=max(1,float(distance.max())*vertical[row]/10)
            hidden[row,extra[distance>limit]]=False
        n,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(hidden))
        for component in range(1,n):
            a,c,d,e,area=stats[component]
            if c>y+(b-y)*.5 and e<(b-y)*.1 and area<(r-x)*(b-y)*.003:
                hidden[labels==component]=False
        if hidden.any():
            # Mirroring is a deterministic donor from the same garment, not
            # the neighbouring sleeve occupying those pixels in the painting.
            pixels=rgb.copy()
            # Where symmetry has no visible donor (asymmetric sleeves), use
            # the closest own-texture pixel rather than copying the occluder.
            donor=m>0
            if config.get('complete_overlap'):
                # Ownership along touching, similarly coloured coats is
                # uncertain by a pixel or two. Never magnify that boundary
                # into a whole filled sleeve; sample safely inside own cloth.
                donor=cv2.erode(np.uint8(donor),np.ones((9,9),np.uint8)).astype(bool)
            _,nearest=cv2.distanceTransformWithLabels(np.uint8(~donor),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
            source_y,source_x=np.where(donor)
            nearest=np.clip(nearest-1,0,len(source_y)-1)
            donor_y=source_y[nearest];donor_x=source_x[nearest]
            mirrored=hidden&reflected&donor[yy,reflected_x]
            pixels[hidden]=rgb[donor_y[hidden],donor_x[hidden]]
            pixels[mirrored]=rgb[yy[mirrored],reflected_x[mirrored]]
            if config.get('complete_overlap'):
                # Nearest-pixel extension makes repeated horizontal stripes
                # at fold boundaries. Inpaint the narrow occlusion using only
                # this garment's safe interior as boundary conditions.
                sy,sx=np.where(donor|hidden)
                l,t=max(0,sx.min()-12),max(0,sy.min()-12)
                rr,bb=min(w,sx.max()+13),min(h,sy.max()+13)
                fill=cv2.inpaint(rgb[t:bb,l:rr].copy(),np.uint8(~donor[t:bb,l:rr])*255,5,cv2.INPAINT_TELEA)
                hm=(hidden&~mirrored)[t:bb,l:rr]
                pixels[t:bb,l:rr][hm]=fill[hm]
            masks.images[iid]=pixels
            masks[iid][hidden]=255
            # Completion must not grow a narrow tab at an asymmetric hem.
            # Smooth only invented pixels; every owned source pixel survives.
            smooth=cv2.morphologyEx(masks[iid],cv2.MORPH_OPEN,np.ones((11,11),np.uint8))
            if config.get('complete_overlap'):
                rounded=cv2.GaussianBlur(masks[iid],(21,21),5)>127
                smooth=np.uint8((smooth>0)&rounded)*255
            if not substantial:
                masks[iid][hidden&(smooth==0)]=0
    masks.visible=visible
    # Remove cloth and its cast shadows, preserving every rack/furniture pixel.
    removal=np.uint8(union)*255
    removal=cv2.dilate(removal,np.ones((23,23),np.uint8))
    # Entire empty rack bays have clean donor support. Clear the bay as a
    # whole so the shadow inside a hanger opening cannot float on the plate.
    for x,y,r,b in config.get('clear_wall',[]):
        removal[y:b,x:r]=255
    removal[structure>0]=0
    # Wide feather avoids rectangular seams while keeping reconstructed wall
    # opaque under the silhouette and its contact shadow.
    distance=cv2.distanceTransform(255-removal,cv2.DIST_L2,3)
    alpha=np.clip(1-distance/14,0,1);alpha[structure>0]=0
    post_distance=cv2.distanceTransform(np.uint8(~union),cv2.DIST_L2,3)
    alpha[post_region]*=np.clip(1-post_distance[post_region]/6,0,1)
    if config.get('clear_wall'):
        # Sleeve shadows also fall on the posts. A distance-to-cloth cap
        # leaves a cuff-shaped ghost there; remove neutral darkening against
        # the donor post profile while retaining its exposed painted grain.
        post_ratio=(rgb.astype(float)+1)/(background.astype(float)+1)
        post_shadow=post_region&(np.ptp(post_ratio,axis=2)<.10)&(post_ratio.mean(2)<.98)&(post_ratio.mean(2)>.55)
        near_cloth=cv2.dilate(np.uint8(union),np.ones((71,71),np.uint8)).astype(bool)
        alpha[post_shadow&near_cloth]=1
        alpha[structure>0]=0
    alpha[hooks]=0
    plate=np.uint8(rgb*(1-alpha[:,:,None])+background*alpha[:,:,None])
    # Apply the same strict rest/removal checks to the completed layers that
    # the exporter uses. Coverage alone cannot establish safe layered motion.
    comp=plate.astype(float).copy()
    for iid in masks.order:
        a=cv2.GaussianBlur(masks[iid],(3,3),.55).astype(float)/255
        comp=comp*(1-a[:,:,None])+masks.images.get(iid,rgb)*a[:,:,None]
    ratio=(rgb.astype(float)+1)/(plate.astype(float)+1)
    shadow=(ratio.max(2)<.985)&(ratio.min(2)>.35)&(np.ptp(ratio,axis=2)<.10)
    lost=(np.linalg.norm(comp-rgb,axis=2)>30)&~shadow
    masks.metrics={}
    masks.background_metric=bool(config.get('pale_cloth'))
    for iid,(x,y,r,b) in boxes.items():
        dx,dy=round((r-x)*.15),round((b-y)*.15)
        region=lost[max(0,y-dy):min(h,b+dy),max(0,x-dx):min(w,r+dx)]
        error=float(region.mean())
        m=masks[iid]>0
        pixels=masks.images.get(iid,rgb)
        unchanged=(np.linalg.norm(pixels.astype(float)-plate.astype(float),axis=2)<14)&m
        wall_fraction=float(unchanged.sum()/max(1,m.sum()))
        if config.get('pale_cloth'):
            wall_fraction=attached_background_fraction(pixels,plate,masks[iid])
        rows=np.flatnonzero((masks[iid]>0).sum(1)>8)
        masks.metrics[iid]={'rest_err':round(error,4),'unchanged_fraction':round(wall_fraction,4),
                            'completed_pixels':int(np.count_nonzero(m&~(visible[iid]>0)))}
        if error>.01:reasons[iid].append('rest check: lost garment or damaged rack')
        if wall_fraction>.03:reasons[iid].append('removal check: wall or unchanged foreground in layer')
        if not len(rows) or rows[-1]<b-12 or rows[0]>y+5:
            reasons[iid].append('silhouette check: missing hanger or garment extremity')
    # A failed neighbouring item must not leave a hole in the conservative
    # painting. Fail the configured rack closed rather than approving layers
    # over a plate whose rejected garments have been removed.
    masks.preview_plate=plate.copy()
    if any(reasons.values()):
        for iid in reasons:
            if not reasons[iid]:reasons[iid]=['rack integrity check: neighbouring garment rejected']
        plate=rgb.copy()
    return plate,masks,reasons
