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
    if config.get('boxes_include_hooks'):
        # Some layouts start at the metal hook, above the rail. The moving
        # silhouette starts at the wooden hanger instead; using the hook y
        # both admits wall above the hanger and falsely fails the top gate.
        red,green,blue=rgb.astype(int).transpose(2,0,1)
        adjusted={}
        for iid,(x,y,r,b) in boxes.items():
            rail=min(config['rails'],key=lambda z: abs(y-z[3])*4+(0 if z[0]<(x+r)/2<z[2] else w))
            warm=(red-blue>48)&(red-green>24)&(rgb.mean(2)<200)
            warm&=(np.abs(xx-(x+r)/2)<20)&(yy>rail[3]+5)&(yy<rail[3]+55)
            rows=np.flatnonzero(warm.sum(1)>=3)
            adjusted[iid]=(x,int(rows[0]) if len(rows) else rail[3]+12,r,b)
        boxes=adjusted
    occupied=np.zeros((h,w),np.uint8)
    for x,y,r,b in boxes.values():
        occupied[max(0,y-3):min(h,b+15),max(0,x-4):min(w,r+4)]=1
    wall=wall_model(rgb,config,occupied)
    if config.get('directional_light'):
        # Window light is a diagonal donor profile, not part of the broad
        # plaster polynomial. Shift clean columns along its projection and
        # fit attenuation only on exposed wall, then continue it behind cloth.
        light=config['directional_light']
        lx,ly,lr,lb=light['bbox'];a,c=light['donor_x']
        slope=light['slope'];ref=(a+c-1)/2
        residual=rgb.astype(float)-wall.astype(float)
        donor_profiles=[]
        for col in range(a,c):
            rows=np.arange(h)+slope*(col-ref)
            donor_profiles.append(np.stack([
                np.interp(rows,np.arange(h),residual[:,col,ch])
                for ch in range(3)],axis=1))
        profile=np.median(donor_profiles,axis=0)
        profile=np.maximum(profile,0)
        clean=cv2.dilate(occupied,np.ones((31,31),np.uint8))==0
        for sx,sy,sr,sb in config['structure']:clean[sy:sb,sx:sr]=False
        amplitudes=[]
        projected=[]
        for col in range(lx,lr):
            donor=np.stack([np.interp(np.arange(h)-slope*(col-ref),np.arange(h),profile[:,ch]) for ch in range(3)],axis=1)
            sel=clean[:,col]&(np.arange(h)>=ly)&(np.arange(h)<lb)&(donor.mean(1)>10)
            projected.append(donor)
            amplitudes.append(np.clip((donor[sel]*residual[sel,col]).sum()/max(1,(donor[sel]**2).sum()),0,1.3) if sel.sum()>12 else np.nan)
        amplitudes=np.array(amplitudes);valid=np.flatnonzero(np.isfinite(amplitudes))
        amplitudes=np.interp(np.arange(lr-lx),valid,amplitudes[valid]) if len(valid) else np.zeros(lr-lx)
        amplitudes=cv2.GaussianBlur(amplitudes[None,:],(31,1),7)[0]
        for col,donor,amplitude in zip(range(lx,lr),projected,amplitudes):
            wall[ly:lb,col]=np.uint8(np.clip(wall[ly:lb,col]+donor[ly:lb]*amplitude,0,255))
    background=wall.copy()
    # Wall-mounted rails cast a continuous shadow behind the hanger apex.
    # A wall polynomial erases it; retaining a rectangular rail band instead
    # preserves hanger fragments. Rebuild only that shadow from clean gaps.
    for band in config.get('wall_profiles', []):
        x,y,r,b=band['bbox']
        donors=np.concatenate([rgb[y:b,a:c] for a,c in band['donor_x']],axis=1)
        profile=np.median(donors,axis=1)
        # Preserve the fitted wall's lateral illumination, using the donor
        # strip only for the rail's vertical shadow profile.
        donor_wall=np.concatenate([wall[y:b,a:c] for a,c in band['donor_x']],axis=1)
        offset=profile-np.median(donor_wall,axis=1)
        offset-=np.median(offset[-5:],axis=0)
        offset=cv2.GaussianBlur(offset,(1,9),1.6)
        fade_y=np.clip((b-1-np.arange(y,b))/8,0,1)
        fade_x=np.minimum(np.clip((np.arange(x,r)-x)/16,0,1),np.clip((r-1-np.arange(x,r))/16,0,1))
        background[y:b,x:r]=np.uint8(np.clip(wall[y:b,x:r].astype(float)+offset[:,None,:]*fade_y[:,None,None]*fade_x[None,:,None],0,255))
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
    # Shirts can reach the shelf top. Rebuild its horizontal painted grain
    # from exposed gaps, rather than filling the hidden wood with wall colour.
    # Opt-in geometry leaves every existing rack's reconstruction unchanged.
    for surface in config.get('surfaces', []):
        x,y,r,b=surface['bbox']
        donors=np.concatenate([rgb[y:b,a:c] for a,c in surface['donor_x']],axis=1)
        background[y:b,x:r]=np.median(donors,axis=1).astype(np.uint8)[:,None,:]
    # Wall rails can be crossed by a long garment. Preserve structure only
    # where its paint agrees with clean rail donors; otherwise it is cloth,
    # not an immutable rectangular piece of the original painting.
    rebuilt_structure=np.zeros((h,w),bool)
    for rail in config.get('rebuild_rails', []):
        x,y,r,b=rail['bbox']
        donors=np.concatenate([rgb[y:b,a:c] for a,c in rail['donor_x']],axis=1)
        profile=np.median(donors,axis=1).astype(np.uint8)
        background[y:b,x:r]=profile[:,None,:]
        rebuilt_structure[y:b,x:r]=True
    for furniture in config.get('furniture_donors', []):
        x,y,r,b=furniture['bbox'];a,c,d,e=furniture['donor_bbox']
        donor=rgb[c:e,a:d]
        if furniture.get('flip_x'):donor=donor[:,::-1]
        donor=cv2.resize(donor,(r-x,b-y),interpolation=cv2.INTER_LINEAR)
        if furniture.get('wood_donor_only'):
            # A rectangular furniture donor includes old wall beside its
            # sloped top. Transfer wood, not that wall's edge or illumination.
            colour=donor.astype(int)
            keep=(colour[:,:,0]-colour[:,:,2]>25)&(colour[:,:,0]-colour[:,:,1]>8)
            keep=cv2.morphologyEx(np.uint8(keep),cv2.MORPH_CLOSE,np.ones((5,5),np.uint8)).astype(bool)
            keep |= np.arange(y,b)[:,None]>=furniture['floor_y']
            background[y:b,x:r][keep]=donor[keep]
        else:background[y:b,x:r]=donor
    structure=np.zeros((h,w),np.uint8)
    for x,y,r,b in config['structure']:
        structure[y:b,x:r]=1
    for x,y,r,b in config.get('wood_structure', []):
        # Round rail end caps must not preserve a square of the old wall.
        patch=rgb[y:b,x:r].astype(int)
        wood=(patch[:,:,0]-patch[:,:,2]>25)&(patch[:,:,0]-patch[:,:,1]>10)
        structure[y:b,x:r]|=cv2.dilate(np.uint8(wood),np.ones((3,3),np.uint8))
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
    if config.get('exclude_post_wood'):
        # Broad rack posts can include a second, vertical rear upright. Their
        # warmer grain exceeds the old narrow blue-channel interval; do not
        # classify that exposed wood as a shirt sleeve or a loose side strip.
        wood=post_region&(red-green>12)&(green-blue>8)&(red-blue>24)
    wood=cv2.dilate(np.uint8(wood),np.ones((3,3),np.uint8)).astype(bool)&post_region
    surface_wood=np.zeros((h,w),bool)
    for surface in config.get('surfaces', []):
        # Cream hems are warm too; a floor band has no exposed wood to veto.
        if surface.get('cloth_over_surface'):continue
        x,y,r,b=surface['bbox']
        # The ROI includes the hem's contact shadow and can otherwise attach
        # a long strip of shelf edge to a pale shirt. Warm exposed wood is
        # surface, while the neutral cloth continues to its painted hem.
        surface_wood[y:b,x:r]=(red-green>12)[y:b,x:r]&(green-blue>8)[y:b,x:r]
    if config.get('rebuild_rails'):
        structure[rebuilt_structure&(delta>25)]=0
    fg=(delta>19)&~shadows&~structure.astype(bool)
    conservative_fg=fg.copy()
    if config.get('neutral_garments'):
        # Grey woven shirts can be a uniform darkening of plaster too.
        # Let the local material GMM distinguish them from cast shadows;
        # a global multiplicative-shadow veto removes whole pale sleeves.
        fg=(delta>19)&~structure.astype(bool)
    masks=GarmentMasks(); reasons={}
    costs=[]; candidates=[]
    hooks=np.zeros((h,w),bool)
    for iid,(x,y,r,b) in boxes.items():
        # Two stacked rows can share the same horizontal centre. Select the
        # rail above the garment by vertical distance, not centre-x alone.
        rail=min(config['rails'],key=lambda z: abs(y-z[3])*4 + (0 if z[0]< (x+r)/2 < z[2] else w))
        top=rail[3]+(1 if config.get('retain_hanger_wood') else 3)
        # Metal hooks stay with the rack; the entire wooden hanger moves.
        # Find the first broad warm wood row, not a bbox-dependent stem cut.
        warm=(red-blue>48)&(red-green>24)&(rgb.mean(2)<200)&(np.abs(xx-(x+r)/2)<20)
        warm&=(yy>=rail[3])&(yy<y+8)
        woodrows=np.flatnonzero((warm.sum(1)>=3)&((warm.sum(1)<25) if config.get('retain_hanger_wood') else True))
        hanger_top=int(woodrows[0]) if len(woodrows) else top+6
        apex=np.flatnonzero(warm[hanger_top])
        hook_x=float(np.median(apex)) if len(apex) else (x+r)/2
        hanger_top-=config.get('hanger_top_margin',0)
        hooks|=(np.abs(xx-hook_x)<=config.get('hook_radius',3))&(yy>=rail[3]-1)&(yy<hanger_top)
        roi=(xx>=x-8)&(xx<r+8)&(yy>=top)&(yy<b+10)
        # Local GrabCut separates pale cuffs from wall and retains fine hanger
        # edges without turning the triangular hanger opening into opaque wall.
        seeds=np.zeros((h,w),np.uint8)
        seeds[roi]=cv2.GC_PR_BGD
        material=rgb[max(y,0):min(b,h),max(0,round(x+(r-x)*.3)):min(w,round(r-(r-x)*.3))]
        pale=config.get('neutral_garments') and (np.median(material.mean(2))>120 or config.get('preserve_pale_details'))
        # Dark denim can have pale rolled cuffs outside its central material
        # sample. A shadow veto drops those cuffs before GrabCut sees them.
        local_fg=fg if pale else conservative_fg
        seeds[roi&local_fg]=cv2.GC_PR_FGD
        core=(xx>x+(r-x)*.25)&(xx<r-(r-x)*.25)&(yy>y+(b-y)*.15)&(yy<b-(b-y)*.12)&fg
        seeds[core&local_fg]=cv2.GC_FGD
        if pale:
            # Painted hems can match plaster exactly. Closed outline contours
            # provide interior seeds, not a filled cutout: erode well inside
            # the outline so soft cast-shadow edges never become foreground.
            patch=rgb[y:min(h,b+4),x:r]
            edges=cv2.Canny(patch,35,90)
            edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            interior=np.zeros(edges.shape,np.uint8)
            for contour in contours:
                if cv2.contourArea(contour)>(r-x)*(b-y)*.15:
                    cv2.drawContours(interior,[contour],-1,1,-1)
            interior=cv2.erode(interior,np.ones((15,15),np.uint8))
            interior[:round((b-y)*.12)]=0
            seeds[y:min(h,b+4),x:r][interior>0]=cv2.GC_FGD
        seeds[structure>0]=cv2.GC_BGD
        if config.get('preserve_leg_gaps'):
            # The shadow between trouser legs is enclosed by the two hems
            # in some painted outlines. Pale-cloth contour seeds otherwise
            # teach GrabCut that this smooth, neutral shadow is textile.
            # Follow only smooth wall-colour pixels connected to below the
            # hems; a dark fold inside the garment is not a leg opening.
            value=rgb.astype(np.float32)
            mean=cv2.blur(value,(3,3))
            variation=np.sqrt(np.maximum(0,cv2.blur(value*value,(3,3))-mean*mean).mean(2))
            gap=(np.ptp(ratio,axis=2)<.10)&(ratio.mean(2)>.35)&(ratio.mean(2)<1.03)&(variation<3)
            gap &= (np.abs(xx-(x+r)/2)<(r-x)*.20)&(yy>y+(b-y)*.42)&(yy<b+12)
            _,labels=cv2.connectedComponents(np.uint8(gap))
            bottom=labels[min(h-1,b+5),max(0,x):min(w,r)]
            connected=np.isin(labels,np.unique(bottom[bottom>0]))&gap
            # A bbox-centre foreground seed is appropriate for shirts, but
            # lands on wall between split legs. Let the material GMM decide
            # this band instead of forcing its cast shadow into the layer.
            split=(np.abs(xx-(x+r)/2)<(r-x)*.23)&(yy>y+(b-y)*.40)&roi
            seeds[split&(seeds==cv2.GC_FGD)]=cv2.GC_PR_FGD
            seeds[connected]=cv2.GC_BGD
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
            # Trouser labels can match plaster at the waistband. The wall
            # veto belongs above the cloth, not eight pixels into its top.
            cloth_top=y if config.get('preserve_waistband') else y+8
            m[((delta<15)|shadows)&(yy<cloth_top)]=0
        else:
            m[(delta<15)|shadows]=0
        n,lab,stats,_=cv2.connectedComponentsWithStats(m)
        for k in range(1,n):
            if stats[k,4]<8:m[lab==k]=0
        if config.get('symmetric_sleeves') and n>1:
            # Remove neighbour trim before dilation: dilation can bridge a
            # detached dark seam across the wall gap onto the real sleeve.
            main=1+np.argmax(stats[1:,4])
            for component in range(1,n):
                a,c,d,e,area=stats[component]
                hanger=c<y+8 and abs(a+d/2-(x+r)/2)<(r-x)*.3
                if component!=main and not hanger:m[lab==component]=0
        m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        m=cv2.dilate(m,np.ones((3,3),np.uint8))
        m[~roi|(structure>0)|(yy<hanger_top)|surface_wood]=0
        if config.get('preserve_neck_openings') and np.median(material.mean(2))<170:
            # Dark bodices can enclose a grey wall opening below the hanger.
            # It is neither a cloth panel nor a hole to fill with opaque RGB.
            neck=(np.abs(xx-hook_x)<(r-x)*.23)&(yy<min(y+(b-y)*.18,y+26))&(yy>hanger_top)
            wall_neck=neck&(np.ptp(rgb.astype(int),axis=2)<12)&(rgb.mean(2)>180)&(delta<40)
            m[wall_neck]=0
        if config.get('retain_hanger_wood'):
            # A cloth-trained GMM discards the differently coloured wooden
            # hanger. Its warm paint is strong evidence only in the narrow
            # band between the selected rail and the garment shoulders.
            hanger=(red-blue>30)&(red-green>10)&(rgb.mean(2)<225)
            hanger &= roi&(yy>=hanger_top)&(yy<y+14)&~structure.astype(bool)
            hanger=cv2.morphologyEx(np.uint8(hanger),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            hanger=cv2.dilate(hanger,np.ones((3,3),np.uint8))
            m[(hanger>0)&roi&(delta>10)&(yy>=hanger_top)]=1
        if config.get('hanger_crossbars') and (not config.get('retain_hanger_wood') or y-rail[3]>36):
            # Thin silver trouser-hanger crossbars can match the wall and
            # disappear under the above-waist shadow veto. Retain detected
            # horizontal edges only, leaving the hanger opening transparent.
            gray=cv2.GaussianBlur(rgb.mean(2).astype(np.float32),(3,3),.6)
            rows=np.arange(max(top+1,y-8),min(h-1,y+1))
            l=round(x+(r-x)*.25);rr=round(r-(r-x)*.25)
            contrast=np.median(np.abs(gray[rows+1,l:rr]-gray[rows-1,l:rr]),axis=1)
            strong=rows[contrast>8]
            if len(strong):
                row=int(strong[0])
                waist=np.flatnonzero(m[min(h-1,y+5)])
                waist=waist[(waist>=x)&(waist<r)]
                if len(waist):
                    samples=[]
                    for col in range(int(waist[0]),int(waist[-1])+1):
                        nearby=np.arange(max(top+1,row-2),min(h-1,row+3))
                        edges=nearby[np.abs(gray[nearby+1,col]-gray[nearby-1,col])>8]
                        if len(edges):samples.append((col,int(edges[0])))
                    if len(samples)>8:
                        samples=np.array(samples)
                        line=np.polyfit(samples[:,0],samples[:,1],1)
                        for col in range(int(waist[0]),int(waist[-1])+1):
                            edge=int(round(np.polyval(line,col)))
                            m[edge:edge+4,col]=1
        # Ownership in overlapping boxes follows material sampled from the
        # unoccluded centre; the spatial term only breaks similar-colour ties.
        if config.get('recover_warm_trim'):
            # Cream binding on black cloth is absent from its central material
            # sample. Restore its warm pixels next to the detected outline;
            # neutral plaster and cast shadows cannot satisfy this evidence.
            near=cv2.dilate(m,np.ones((9,9),np.uint8))>0
            trim=near&roi&(red-blue>15)&(rgb.mean(2)>135)&(delta>15)
            m[trim&~structure.astype(bool)&(yy>=hanger_top)]=1
        colors=np.float32(rgb[core])
        cv2.setRNGSeed(23)
        _,_,centers=cv2.kmeans(colors, min(5,len(colors)),None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,.5),3,cv2.KMEANS_PP_CENTERS)
        colorcost=np.min([np.linalg.norm(rgb.astype(float)-c,axis=2) for c in centers],axis=0)
        upper=(m>0)&(yy<y+9)&(yy>top)
        axis=float(np.median(xx[upper])) if upper.any() else (x+r)/2
        mirror_x=np.clip(np.rint(2*axis-xx).astype(int),0,w-1)
        symmetry=np.linalg.norm(rgb.astype(float)-rgb[yy,mirror_x].astype(float),axis=2)
        weight=np.where((b-y>2*(r-x))&(yy>y+(b-y)*.5),.7,.35)
        if config.get('symmetric_sleeves'):
            # Sleeve seams are often the same trim colour on neighbouring
            # shirts. The matching opposite sleeve disambiguates that trim.
            weight=np.where((yy>y+(b-y)*.12)&(yy<y+(b-y)*.57),.85,weight)
        cost=colorcost*(1-weight)+symmetry*weight+12*np.abs(xx-(x+r)/2)/max(1,r-x)
        # Reject exposed wood only when the opposite sleeve confirms a
        # different material; warm pink/ochre cloth can share wood's hue.
        exposed=wood&(symmetry>45)&(colorcost>30)
        if config.get('deep_overlap'):
            # Patterned collars can share the post's warm hue. Require
            # agreement with the donor wood and stay below the collar.
            exposed &= (np.linalg.norm(rgb.astype(float)-background.astype(float),axis=2)<35)&(yy>y+(b-y)*.2)
        if config.get('exclude_post_wood'):
            exposed |= wood & (yy>y+12)
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
    if config.get('symmetric_sleeves') or config.get('ownership_graphcut'):
        # Palette-only ownership can alternate along a shared sleeve seam.
        # A local graph cut keeps ownership on the painted material boundary;
        # definite seeds come from each garment outside the overlapping band.
        entries=list(boxes.items())
        for left in range(len(entries)):
            for right in range(left+1,len(entries)):
                lb=entries[left][1];rb=entries[right][1]
                if lb[0]>rb[0]:continue
                shared=(candidates[left]|candidates[right])
                shared &= (xx>=rb[0]-8)&(xx<lb[2]+8)
                shared &= (yy>=max(lb[1],rb[1]))&(yy<min(lb[3],rb[3]))
                sy,sx=np.where(shared)
                if not len(sx):continue
                x0=max(0,int(sx.min())-45);x1=min(w,int(sx.max())+46)
                y0=max(0,int(sy.min())-12);y1=min(h,int(sy.max())+13)
                own=owner[y0:y1,x0:x1]
                seed=np.where(own==left,cv2.GC_PR_FGD,cv2.GC_PR_BGD).astype('uint8')
                lc=candidates[left][y0:y1,x0:x1];rc=candidates[right][y0:y1,x0:x1]
                band=shared[y0:y1,x0:x1]
                seed[lc&~band]=cv2.GC_FGD
                seed[rc&~band]=cv2.GC_BGD
                if not (seed==1).any() or not (seed==0).any():continue
                cv2.setRNGSeed(23)
                cv2.grabCut(rgb[y0:y1,x0:x1].copy(),seed,None,np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_MASK)
                selection=shared[y0:y1,x0:x1]
                own[selection&((seed==1)|(seed==3))]=left
                own[selection&((seed==0)|(seed==2))]=right
    # A dark edge on the neighbouring coat can match this garment's palette.
    # Reattach thin row fragments to the adjacent broad silhouette; do not
    # let a matching black cuff/hem become a long loose strip on another coat.
    for k,(iid,(x,y,r,b)) in enumerate(boxes.items()):
        if config.get('preserve_overlap_tips'):continue
        # Short shirts overlap at their sleeves, above the ordinary hem band.
        start=.15 if config.get('symmetric_sleeves') else (.25 if b-y>2*(r-x) else .5)
        for row in range(max(0,round(y+(b-y)*start)),min(h,b+10)):
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
    if config.get('solid_overlap_front'):
        # A pale print on a dark front cuff can match the rear garment's
        # palette. An island enclosed by the front silhouette on both sides
        # belongs to that cuff; do not leave a print-shaped hole in it.
        ids=list(boxes)
        depth_order=config.get('depth_order',ids)
        rank=np.array([depth_order.index(iid) for iid in ids])
        candidate_count=np.sum(candidates,axis=0)
        colour=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV) if config.get('front_colour_edges') else None
        for iid in depth_order:
            k=ids.index(iid)
            shared=candidates[k]&(candidate_count>1)
            if not shared.any():continue
            if config.get('front_colour_edges'):
                # Colour antialiasing can put one blue cuff pixel outside
                # the graph-cut seam. Recover only the front material's
                # saturated hue next to its silhouette, never beige wall.
                x,y,r,b=boxes[iid]
                sample=colour[y+(b-y)//4:b-(b-y)//4,x+(r-x)//3:r-(r-x)//3]
                strong=sample[:,:,1]>65
                if strong.mean()>.3:
                    hues=sample[:,:,0][strong]
                    hue=int(np.bincount(hues,minlength=180).argmax())
                    delta_h=np.abs(colour[:,:,0].astype(int)-hue)
                    match=(np.minimum(delta_h,180-delta_h)<12)&(colour[:,:,1]>65)
                    near=cv2.dilate(np.uint8(union&(owner==k)),np.ones((5,5),np.uint8))>0
                    owner[shared&near&match&(rank[owner]<rank[k])]=k
            for row in np.flatnonzero(shared.any(1)):
                run=np.flatnonzero(union[row]&(owner[row]==k))
                if len(run):
                    fill=shared[row]&(xx[row]>=run[0])&(xx[row]<=run[-1])&(rank[owner[row]]<rank[k])
                    owner[row,fill]=k
    for k,(iid,box) in enumerate(boxes.items()):
        m=np.uint8(union&(owner==k))*255
        # Detached pieces in an adjacent bbox are not part of this garment.
        # The hanger and real drawcords remain attached to the main silhouette.
        n,labels,stats,_=cv2.connectedComponentsWithStats(m)
        open_textile=config.get('open_textiles') and box[3]-box[1]>2.3*(box[2]-box[0])
        if n>1 and not open_textile:
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
            if area<(r-x)*(b-y)*.15 and c>y+8 and a>x and a+d<r and c+e<b and not config.get('preserve_leg_gaps') and not open_textile and (not config.get('preserve_neck_openings') or c>y+(b-y)*.25):
                free=(lab==j)&~union;m[free]=255
        # Thin post-grain remnants are not sleeve details. Regularise only
        # the known post band, leaving drawcords and garment hems elsewhere.
        # A sleeve crossing a narrow post needs the smaller opening: the
        # broad-post cleanup would round off its real pointed cuff.
        post_size=config.get('post_opening',21 if config.get('exclude_post_wood') else 7)
        post_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(post_size,post_size)) if config.get('exclude_post_wood') else np.ones((7,7),np.uint8)
        opened=cv2.morphologyEx(m,cv2.MORPH_OPEN,post_kernel)
        if not config.get('deep_overlap'):
            m[post_region]=opened[post_region]
        if config.get('surfaces'):
            # A neutral antialiased shelf lip survives the wood-colour test.
            # Remove those one-pixel horizontal spurs only on the surface;
            # the broad cloth hem and all sleeve/hanger detail stay intact.
            clean=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
            for surface in config['surfaces']:
                sx,sy,sr,sb=surface['bbox']
                m[sy:sb,sx:sr]=clean[sy:sb,sx:sr]
        if config.get('trim_hem_shadows') and np.median(material.mean(2))<120:
            # Pale rolled cuffs and their contact shadows share a palette.
            # Follow the last painted hem edge, removing only smooth neutral
            # darkening below it; do not round off the cuff or its stitching.
            gray=cv2.GaussianBlur(rgb.mean(2).astype(np.float32),(3,3),.6)
            value=rgb.astype(np.float32)
            mean=cv2.blur(value,(3,3))
            variation=np.sqrt(np.maximum(0,cv2.blur(value*value,(3,3))-mean*mean).mean(2))
            for col in range(max(0,x),min(w,r)):
                rows=np.flatnonzero(m[:,col])
                if not len(rows):continue
                end=int(rows[-1])
                lo=max(1,end-18,y+round((b-y)*.85))
                edge=np.arange(lo,min(h-1,end))
                edge=edge[(gray[edge+1,col]-gray[edge-1,col])>30]
                if not len(edge):continue
                hem=int(edge[-1])
                cloth=rgb[max(lo,hem-8):max(lo+1,hem-2),col].astype(float)
                if cloth.mean()>110 and np.median(np.ptp(cloth,axis=1))<25:
                    # At a pale cuff, the outline itself is stronger evidence
                    # than the GMM: keep the entire cloth up to that edge and
                    # discard all contact shadow below, including its border.
                    owned=rows[rows<=hem+1]
                    if len(owned) and hem-owned[-1]<12:
                        m[int(owned[-1]):hem+2,col]=1
                        m[hem+2:end+1,col]=0
                    continue
                tail=np.arange(hem+2,end+1)
                neutral=(np.ptp(ratio[tail,col],axis=1)<.08)&(ratio[tail,col].mean(1)>.6)
                m[tail[neutral&(variation[tail,col]<3)],col]=0
        if open_textile:
            # Open lace is a network of threads and embroidery, not an
            # opaque wall-coloured skirt. Keep its detached stitches; a
            # largest-component filter or hole fill destroys this material.
            local=cv2.medianBlur(rgb,25)
            local_delta=np.linalg.norm(rgb.astype(float)-local.astype(float),axis=2)
            detail=local_delta>10
            pink=(red-green>12)&(blue-green>-12)
            peach=(red-green>12)&(red-blue>18)&(rgb.mean(2)>130)
            lace=(detail|pink|peach)&(rgb.mean(2)>100)
            region=(xx>=x-8)&(xx<r+8)&(yy>y+(b-y)*.58)&(yy<b+10)
            # Grow only onto the thread's antialiased edge, not onto smooth
            # wall between stitches. Unconditional dilation makes opaque lace.
            lace=(cv2.dilate(np.uint8(lace),np.ones((3,3),np.uint8))>0)&((local_delta>4)|pink|peach)
            for furniture in config.get('furniture_donors', []):
                sx,sy,sr,sb=furniture['bbox']
                exposed_wood=(red-green>22)&(red-green<70)&(green-blue>15)&(red<180)&(green<125)&(blue<100)
                lace[sy:sb,sx:sr]&=~exposed_wood[sy:sb,sx:sr]
            # Long straight neutral edges belong to the underlying furniture,
            # not to the small embroidered flowers or winding mesh threads.
            for furniture in config.get('furniture_donors', []):
                sx,sy,sr,sb=furniture['bbox']
                patch=np.uint8(lace[sy:sb,sx:sr]&(np.ptp(rgb[sy:sb,sx:sr].astype(int),axis=2)<24))
                straight=cv2.morphologyEx(patch,cv2.MORPH_OPEN,np.ones((1,25),np.uint8))
                straight|=cv2.morphologyEx(patch,cv2.MORPH_OPEN,np.ones((25,1),np.uint8))
                lace[sy:sb,sx:sr]&=~(straight>0)
            lace &= ~(union&(owner!=k))
            m[region]=np.uint8(lace[region])*255
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
        if config.get('symmetric_sleeves'):
            # Painted shirts are slightly asymmetric. Align the mirrored
            # sleeve to its own visible shoulder before the occlusion starts;
            # hanger-axis reflection alone leaves a step at that boundary.
            for side in (-1,1):
                half=(xx-center)*side>0
                contact=reflected&front&half&(yy>y+12)&(yy<b)
                rows=np.flatnonzero(contact.sum(1)>2)
                if not len(rows):continue
                start=int(rows[0]);offsets=[]
                for row in range(max(y+12,start-28),start-2):
                    own_x=np.flatnonzero((m[row]>0)&half[row])
                    mirror_x=np.flatnonzero(reflected[row]&half[row])
                    if len(own_x)>8 and len(mirror_x)>8:
                        edge=-1 if side==1 else 0
                        offsets.append(own_x[edge]-mirror_x[edge])
                shift=int(np.clip(np.median(offsets),-16,16)) if offsets else 0
                reflected_x[half]=np.clip(reflected_x[half]+shift,0,w-1)
            reflected=m[yy,reflected_x]>0
            # Layout boxes describe visible cloth, so a hidden sleeve can
            # extend beyond the box. Complete it from the opposite sleeve
            # rather than clipping to that visible bbox or bridging row gaps.
            front=cv2.dilate(np.uint8(front),np.ones((7,7),np.uint8)).astype(bool)
            hidden=reflected&front&(yy>y+12)&(yy<b+3)&(m==0)
            if config.get('complete_sleeve_edges'):
                # Both sleeves can be occluded: mirroring then duplicates
                # the inward cut on both sides. Extrapolate the visible outer
                # shoulder and lower cuff edges to their intersection instead.
                # Only complete behind a front garment, using own cloth below.
                for side in (-1,1):
                    samples=[]
                    for row in range(y+1,min(h,b)):
                        run=np.flatnonzero(m[row])
                        if len(run)>30:
                            samples.append((row,run[0] if side<0 else run[-1]))
                    samples=np.array(samples)
                    if not len(samples):continue
                    outer=[]
                    for lo,hi in ((.12,.25),(.48,.55)):
                        q=samples[(samples[:,0]>y+(b-y)*lo)&(samples[:,0]<y+(b-y)*hi)]
                        if len(q)<6:break
                        outer.append(np.polyfit(q[:,0],q[:,1],1))
                    if len(outer)!=2:continue
                    shoulder,cuff=outer
                    if not (.15<side*shoulder[0]<1.3 and -3<side*cuff[0]<-.4):continue
                    for row in range(round(y+(b-y)*.25),round(y+(b-y)*.56)):
                        edge=(min if side>0 else max)(np.polyval(shoulder,row),np.polyval(cuff,row))
                        own=np.flatnonzero(m[row])
                        if not len(own):continue
                        start,stop=sorted((int(round(edge)),int(own[0] if side<0 else own[-1])))
                        start=max(0,start);stop=min(w,stop+1)
                        hidden[row,start:stop] |= front[row,start:stop] & (m[row,start:stop]==0)
        else:
            hidden=(reflected|envelope)&front&(yy>y+12)&(yy<b+3)&(xx>x-8)&(xx<r+8)
        if config.get('complete_flared_edges'):
            # A flare can be asymmetric about the hanger. Continue its own
            # visible straight side behind the front sleeve, rather than
            # mirroring a narrower opposite edge and leaving a cuff notch.
            near=cv2.dilate(np.uint8(front),np.ones((9,9),np.uint8))>0
            for side in (-1,1):
                contact=[]
                for row in range(y+20,min(h,b)):
                    run=np.flatnonzero(m[row])
                    if len(run)>20:
                        edge=int(run[0] if side<0 else run[-1])
                        if near[row,edge]:contact.append(row)
                if not contact:continue
                start=contact[0];samples=[]
                for row in range(max(y+20,start-40),start-5):
                    run=np.flatnonzero(m[row])
                    if len(run)>20:samples.append((row,int(run[0] if side<0 else run[-1])))
                if len(samples)<12:continue
                sample=np.array(samples);line=np.polyfit(sample[:,0],sample[:,1],1)
                if not .15<side*line[0]<1.0:continue
                for row in range(start,min(h,b)):
                    run=np.flatnonzero(m[row])
                    if not len(run):continue
                    edge=int(round(np.polyval(line,row)))
                    own=int(run[0] if side<0 else run[-1])
                    if side*(edge-own)<=0:continue
                    a,e=sorted((edge,own));a=max(0,a);e=min(w,e+1)
                    hidden[row,a:e] |= near[row,a:e]&(m[row,a:e]==0)
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
            if substantial or config.get('complete_flared_edges') or row<=y+(b-y)*.5 or vertical[row]>=10:continue
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
            if config.get('complete_overlap') and not config.get('symmetric_sleeves'):
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
            if config.get('symmetric_sleeves'):
                # A front sleeve's dark antialiased outline can remain on the
                # rear cutout's edge. Replace that narrow seam from the same
                # rear sleeve as the completion, feathering into visible cloth.
                safe=cv2.erode(np.uint8(m>0),np.ones((5,5),np.uint8)).astype(bool)
                _,labels=cv2.distanceTransformWithLabels(np.uint8(~safe),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
                sy,sx=np.where(safe)
                nearest=np.clip(labels-1,0,len(sy)-1)
                reflected_donor=nearest[yy,reflected_x]
                texture=rgb[sy[reflected_donor],sx[reflected_donor]]
                distance=cv2.distanceTransform(np.uint8(~hidden),cv2.DIST_L2,3)
                seam=(distance<6)&(m>0)
                blend=np.clip((6-distance[seam])/3,0,1)[:,None]
                pixels[hidden]=texture[hidden]
                pixels[seam]=pixels[seam]*(1-blend)+texture[seam]*blend
            masks.images[iid]=pixels
            masks[iid][hidden]=255
            # Completion must not grow a narrow tab at an asymmetric hem.
            # Smooth only invented pixels; every owned source pixel survives.
            smooth=cv2.morphologyEx(masks[iid],cv2.MORPH_OPEN,np.ones((11,11),np.uint8))
            if config.get('complete_overlap'):
                rounded=cv2.GaussianBlur(masks[iid],(21,21),5)>127
                smooth=np.uint8((smooth>0)&rounded)*255
            if not substantial and not config.get('symmetric_sleeves') and not config.get('complete_flared_edges'):
                masks[iid][hidden&(smooth==0)]=0
            if config.get('smooth_overlap_edges'):
                # A one-pixel ownership seam can become a tiny pointed tab
                # when the front sleeve swings away. Regularise only the
                # completed contact boundary, never hems, print or hangers.
                contact=cv2.dilate(np.uint8(hidden),np.ones((17,17),np.uint8))>0
                kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
                clean=cv2.morphologyEx(masks[iid],cv2.MORPH_CLOSE,kernel)
                clean=cv2.morphologyEx(clean,cv2.MORPH_OPEN,kernel)
                added=contact&(clean>0)&(masks[iid]==0)
                pixels[added]=rgb[donor_y[added],donor_x[added]]
                masks[iid][contact]=clean[contact]
    if config.get('trim_sleeve_shadows'):
        # A grey cuff's cast shadow can have the same palette as its cloth.
        # Continue the short, visible diagonal cuff edge to the torso; a
        # shadow's vertical/square outer edge must not become moving fabric.
        for iid,m in masks.items():
            x,y,r,b=boxes[iid]
            for side in (-1,1):
                profile=[]
                for row in range(y+15,min(h,b)):
                    run=np.flatnonzero(m[row])
                    if len(run)>30:profile.append((row,run[0] if side<0 else run[-1]))
                profile=np.array(profile)
                if not len(profile):continue
                search=profile[(profile[:,0]>y+(b-y)*.25)&(profile[:,0]<y+(b-y)*.7)]
                if not len(search):continue
                peak=int(search[np.argmax(side*search[:,1]),0])
                cuff=profile[(profile[:,0]>=peak+4)&(profile[:,0]<peak+18)]
                body=profile[(profile[:,0]>y+(b-y)*.78)&(profile[:,0]<y+(b-y)*.94)]
                if len(cuff)<8 or len(body)<8:continue
                slope,intercept=np.polyfit(cuff[:,0],cuff[:,1],1)
                if not -2.5<side*slope<-.55:continue
                torso=float(np.median(body[:,1]))
                torso_line=np.array([0.,torso])
                material=np.median(rgb[y+(b-y)//3:b-(b-y)//4,x+(r-x)//3:r-(r-x)//3],axis=(0,1))
                if 120<material.mean()<195 and np.ptp(material)<14:
                    # Neutral heather cloth casts a same-colour wall shadow.
                    # Its dark painted side outline, unlike that shadow, is
                    # visible throughout the torso. Fit that line from local
                    # dark-edge donors, rejecting isolated print/buttons.
                    outline=[]
                    gray=rgb.mean(2)
                    for row,edge in profile:
                        if not y+(b-y)*.58<row<y+(b-y)*.78:continue
                        lo=max(0,int(edge)-20);hi=min(w,int(edge)+21)
                        strip=gray[row,lo:hi]
                        pos=int(np.argmin(strip))
                        if np.median(strip)-strip[pos]>30:outline.append((row,lo+pos))
                    if len(outline)>15:
                        outline=np.array(outline)
                        line=np.polyfit(outline[:,0],outline[:,1],1)
                        keep=np.abs(outline[:,1]-np.polyval(line,outline[:,0]))<5
                        if keep.sum()>12:
                            torso_line=np.polyfit(outline[keep,0],outline[keep,1],1)
                            torso_line[1]+=side*2 # retain the antialiased outline
                            for row in range(round(y+(b-y)*.56),b-7):
                                outside=(xx[row]-np.polyval(torso_line,row))*side>0
                                m[row,outside]=0
                for row in range(peak+5,min(b,round(y+(b-y)*.72))):
                    edge=(max if side>0 else min)(np.polyval(torso_line,row),slope*row+intercept)
                    outside=(xx[row]-edge)*side>0
                    m[row,outside]=0
    if config.get('trim_flat_wall'):
        value=rgb.astype(np.float32)
        mean=cv2.blur(value,(5,5))
        variation=np.sqrt(np.maximum(0,cv2.blur(value*value,(5,5))-mean*mean).mean(2))
        plaster=(np.ptp(rgb.astype(int),axis=2)<12)&(rgb.mean(2)>145)&(variation<2.4)
        for iid,m in masks.items():
            x,y,r,b=boxes[iid]
            # Contact shadows below a painted hem are neutral; unlike cream
            # cloth they have no warm pigment, and unlike black cloth they
            # are light. Keep the lower boundary tied to its visible outline.
            hem=(yy>b-3)&(rgb.mean(2)>110)&(rgb.mean(2)<220)&(np.ptp(rgb.astype(int),axis=2)<14)
            hem &= shadows
            m[hem]=0
            m[yy>b+1]=0
            # Flat plaster islands can be left by a source crop around a hem.
            # Only remove connected patches touching the silhouette boundary;
            # enclosed cream fabric panels retain their folds and their area.
            n,lab,stats,_=cv2.connectedComponentsWithStats(np.uint8(plaster&(yy>b-4)&shadows&(m>0)))
            edge=(m>0)&~cv2.erode(np.uint8(m>0),np.ones((3,3),np.uint8)).astype(bool)
            touching=set(np.unique(lab[edge&plaster]))-{0}
            for j in touching:
                if stats[j,4]>=12:m[lab==j]=0
    if any(config.get(k) for k in ('trim_boundary_shadows', 'trim_neutral_hem_shadows', 'trim_smooth_boundary_shadows')):
        # Pale knits and their grey cast shadows can share the GrabCut GMM.
        # Remove smooth, neutral shadow patches only when connected to the
        # cutout edge. Enclosed labels/prints and warm cream yarn survive.
        value=rgb.astype(np.float32)
        mean=cv2.blur(value,(5,5))
        variation=np.sqrt(np.maximum(0,cv2.blur(value*value,(5,5))-mean*mean).mean(2))
        neutral=(np.ptp(value,axis=2)<14)&(value.mean(2)>95)
        shadow_colour=neutral&(np.ptp(ratio,axis=2)<.10)&(ratio.mean(2)>.5)&(ratio.mean(2)<.99)
        smooth_shadow=shadow_colour&(variation<4)
    if config.get('trim_boundary_shadows'):
        for iid,m in masks.items():
            edge=(m>0)&~cv2.erode(np.uint8(m>0),np.ones((3,3),np.uint8)).astype(bool)
            n,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(shadow_colour&(m>0)))
            touching=set(np.unique(labels[edge&shadow_colour]))-{0}
            for j in touching:
                patch=labels==j
                # A thin crop-edge shadow can be narrower than the texture
                # window, so its two borders hide every smooth interior pixel.
                thin=stats[j,4]>=12 and min(stats[j,2],stats[j,3])<=8
                # Heather knit is neutral too, but only a small fraction of
                # its yarn is smooth. Require a zone-configured smooth share
                # before removing a whole connected neutral component.
                smooth_count=np.count_nonzero(patch&smooth_shadow)
                smooth_enough=smooth_count>=12 and smooth_count/max(1,stats[j,4])>=config.get('boundary_shadow_smooth_fraction',0)
                if thin or smooth_enough:m[patch]=0
            # Removing a shadow can detach its antialiased outer outline.
            # Keep the main knit and the wooden hanger, not loose grey arcs.
            n,labels,stats,_=cv2.connectedComponentsWithStats(m)
            if n>1:
                main=1+np.argmax(stats[1:,4]);x,y,r,b=boxes[iid]
                for j in range(1,n):
                    a,c,d,e,area=stats[j]
                    hanger=c<y+8 and abs(a+d/2-(x+r)/2)<(r-x)*.3
                    if j!=main and not hanger:m[labels==j]=0
    if config.get('trim_smooth_boundary_shadows'):
        # Ribbed knit contours can enclose smooth cast-shadow corners in
        # their contour seeds. Test the smooth patch itself, rather than its
        # whole neutral-colour component (which also includes grey yarn).
        for iid,m in masks.items():
            x,y,r,b=boxes[iid]
            boundary_shadow=shadow_colour&(variation<8)&(m>0)
            n,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(boundary_shadow))
            edge=(m>0)&~cv2.erode(np.uint8(m>0),np.ones((3,3),np.uint8)).astype(bool)
            touching=set(np.unique(labels[edge&boundary_shadow]))-{0}
            # Broad smooth cream panels are fabric, not corner shadows.
            limit=np.count_nonzero(m)*.02
            for j in touching:
                if 20<=stats[j,4]<limit:m[labels==j]=0
            # Antialiased shadow borders can remain after their smooth core
            # is gone. Open only neutral shadow pixels below the shoulder;
            # hanger wood and the textured sleeve outline are left intact.
            opened=cv2.morphologyEx(m,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
            m[shadow_colour&(opened==0)&(yy>y+(b-y)*.22)]=0
    if config.get('trim_neutral_hem_shadows'):
        # Heather yarn and its cast shadow are one neutral colour component.
        # Follow the dark painted hem above a smooth shadow tail rather than
        # removing that entire component (which would delete the sweater).
        gray=cv2.GaussianBlur(rgb.mean(2).astype(np.float32),(3,3),.6)
        for iid,m in masks.items():
            x,y,r,b=boxes[iid]
            limits=np.full(w,np.nan)
            for col in range(max(0,x),min(w,r)):
                rows=np.flatnonzero(m[:,col])
                if not len(rows):continue
                end=int(rows[-1])
                if end<y+(b-y)*.65:continue
                tail=np.arange(max(y,end-14),end+1)
                smooth=shadow_colour[tail,col]&(variation[tail,col]<4)
                if smooth.sum()<4:continue
                lo=max(y+1,end-32);edge=np.arange(lo,min(h-2,end-3))
                if not len(edge):continue
                drop=gray[edge+1,col]-gray[edge-1,col]
                strongest=int(np.argmin(drop))
                if drop[strongest]>-16:continue
                hem=int(edge[strongest])
                after=np.arange(hem+1,min(end+1,hem+7))
                hem=int(after[np.argmin(gray[after,col])])
                if np.count_nonzero(smooth&(tail>hem))<4:continue
                limits[col]=hem+2
            # The shadow edge is soft and occasionally misses a column.
            # Median support continues the detected painted hem through
            # those gaps without leaving one-pixel hanging shadow spikes.
            for col in range(max(0,x),min(w,r)):
                nearby=limits[max(0,col-7):min(w,col+8)]
                if np.isfinite(nearby).sum()<5:continue
                hem=int(np.nanmedian(nearby))
                rows=np.flatnonzero(m[:,col])
                if len(rows) and 0<rows[-1]-hem<34:m[hem:int(rows[-1])+1,col]=0
    masks.visible=visible
    # Remove cloth and its cast shadows, preserving every rack/furniture pixel.
    removal=np.uint8(union)*255
    removal=cv2.dilate(removal,np.ones((23,23),np.uint8))
    # Entire empty rack bays have clean donor support. Clear the bay as a
    # whole so the shadow inside a hanger opening cannot float on the plate.
    for x,y,r,b in config.get('clear_wall',[]):
        removal[y:b,x:r]=255
    for surface in config.get('surfaces', []):
        if surface.get('clear_surface'):
            # Floor-contact garments cast shadows beyond their silhouettes,
            # including in the gap between legs. Rebuild the supported surface
            # continuously, rather than leaving islands of the original shadow.
            x,y,r,b=surface['bbox']
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
        if config.get('exclude_post_wood'):
            # Dark cuff outlines cast a deeper, slightly coloured shadow on
            # warm wood than on plaster. Remove that darkening too, without
            # replacing the whole exposed upright and losing its own grain.
            post_shadow |= post_region&(np.ptp(post_ratio,axis=2)<.22)&(post_ratio.mean(2)<.95)&(post_ratio.mean(2)>.25)
            # The antialiased pale cuff edge is brighter than donor wood,
            # rather than a dark shadow. Remove this residual outline too.
            cuff_edge=cv2.dilate(np.uint8(union),np.ones((11,11),np.uint8)).astype(bool)
            post_shadow |= post_region&cuff_edge&(red-green<18)&(rgb.mean(2)>background.mean(2)*1.2)
        near_cloth=cv2.dilate(np.uint8(union),np.ones((71,71),np.uint8)).astype(bool)
        alpha[post_shadow&near_cloth]=1
        alpha[structure>0]=0
    for surface in config.get('surfaces', []):
        if surface.get('clear_surface'):
            # The wall model is invalid below this floor band. Feather inward
            # to the untouched floor, never outward into the rug with plaster.
            x,y,r,b=surface['bbox']
            for row in range(max(y,b-18), min(h,b+14)):
                alpha[row,x:r] *= np.clip((b-row)/surface.get('feather',18),0,1)
            if 'feather' in surface:alpha[b:,x:r]=0
    if config.get('isolate_hooks'):
        # Keeping a rectangular hook ROI also keeps its old wall shadow,
        # producing tiny square patches on the empty rack. Retain only the
        # metal's local contrast, including its antialiased edge and shine.
        local=cv2.medianBlur(rgb,15)
        detail=np.linalg.norm(rgb.astype(float)-local.astype(float),axis=2)>12
        detail=cv2.morphologyEx(np.uint8(detail),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        hooks &= cv2.dilate(detail,np.ones((3,3),np.uint8))>0
    if config.get('retain_hanger_wood'):hooks &= ~((red-blue>30)&(red-green>10))
    alpha[hooks]=0
    # The rail profile also repairs pre-existing crop seams. Reconstruct the
    # whole straight band, including parts formerly covered by a long dress.
    # Keep exposed grain, end caps and screw heads. Donor replacement is
    # needed for cloth crossings and wall-colour gaps in the rail, not for
    # every pixel of an already clean wooden bar.
    crossing=np.zeros((h,w),bool)
    for x,y,r,b in boxes.values():
        crossing |= rebuilt_structure&(xx>=x-8)&(xx<r+8)&(yy>y+14)&(yy<b+10)
    # A pale crop seam can interrupt a rail; dark screw heads and normal
    # painted grain are legitimate rail detail and must survive at rest.
    gaps=rebuilt_structure&(rgb.mean(2)>190)&(background.mean(2)<190)
    source_wood=(red-blue>30)&(red-green>10)
    donor_wood=(background[:,:,0].astype(int)-background[:,:,2]>30)&(background[:,:,0].astype(int)-background[:,:,1]>10)
    repair=crossing|gaps|(rebuilt_structure&source_wood&~donor_wood)
    alpha[rebuilt_structure]=0
    alpha[repair]=1
    for furniture in config.get('furniture_donors', []):
        x,y,r,b=furniture['bbox']
        if furniture.get('repair_occluded_only'):
            # Donor geometry rebuilds hidden furniture, but exposed source
            # grain is exact and must survive the at-rest composite.
            near=cv2.dilate(np.uint8(union),np.ones((15,15),np.uint8))>0
            alpha[y:b,x:r]=near[y:b,x:r]
        else:alpha[y:b,x:r]=1
    for furniture in config.get('furniture_donors', []):
        if not furniture.get('retain_clean_wood'):continue
        x,y,r,b=furniture['bbox']
        # Original warm wood between the stitches is a better grain donor
        # than a resized companion seat. Keep only its material, never the
        # pale/pink embroidery or saturated gold/orange flower centres.
        clean=(red-green>22)&(red-green<70)&(green-blue>15)&(red<180)&(green<125)&(blue<100)
        patch=clean[y:b,x:r]
        background[y:b,x:r][patch]=rgb[y:b,x:r][patch]
        # The donor/source material boundary must not retain a dotted echo
        # of the embroidery's negative space. Regularise only fine grain;
        # the seat edges and legs stay sharp.
        smooth=cv2.medianBlur(background[y:b,x:r],3)
        background[y:b,x:r]=cv2.bilateralFilter(smooth,5,25,3)
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
