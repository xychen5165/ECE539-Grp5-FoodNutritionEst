"""Generate spatial prompts from RGB and original SAM candidates ONLY.

This adapts the existing two-round candidate/point rules to one whole-dish
target: one estimated plate box, up to 16 seed-positive points and up to four
inferred bare-plate negatives. There is no annotation/label input to this module.
"""
import cv2
import numpy as np

PARAMETERS={'points_per_side':16,'pred_iou_thresh':.86,'stability_score_thresh':.92,
            'crop_n_layers':0,'min_mask_region_area':100,'max_positive_points':16,
            'max_negative_points':4,'seed_min_pixels':100,'seed_max_interior_fraction':.65,
            'seed_min_containment':.85,'seed_max_rim_fraction':.10,
            'seed_dedup_iou':.85,'interior_kernel':7,'seed_safety_kernel':11,
            'bare_plate_kernel':9,'box_padding':4}


def fallback(rgb,reason,candidate_count):
    h,w=rgb.shape[:2]
    return {'box_xyxy':[0,0,w-1,h-1],'points_xy':[],'point_labels':[],
            'positive_count':0,'negative_count':0,'plate_candidate_index':None,
            'candidate_count':candidate_count,'seed_count':0,
            'fallback_reason':reason,'reference_mask_used':False}


def derive_prompts(rgb,candidates):
    h,w=rgb.shape[:2]
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV).astype(np.float32)
    options=[]
    for index,candidate in enumerate(candidates):
        mask=candidate['segmentation'].astype(np.uint8)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        hull=cv2.convexHull(max(contours,key=cv2.contourArea))
        silhouette=np.zeros(mask.shape,dtype=np.uint8)
        cv2.drawContours(silhouette,[hull],-1,1,cv2.FILLED)
        x,y,bw,bh=cv2.boundingRect(hull)
        fraction=float(silhouette.mean())
        center_distance=float(np.hypot((x+bw/2)/w-.5,(y+bh/2)/h-.5))
        compactness=float(silhouette.sum())/(bw*bh)
        if not (.12<=fraction<=.85 and .65<=bw/bh<=1.85 and center_distance<.28 and compactness>.60):
            continue
        neutral=float(((hsv[...,1]<60)&(hsv[...,2]>85))[mask>0].mean())
        edges=sum([x<5,y<5,x+bw>w-5,y+bh>h-5])
        score=2*neutral+compactness+fraction-center_distance-.35*edges
        options.append((score,index,silhouette.astype(bool),(x,y,bw,bh)))
    if not options:
        return fallback(rgb,'no_geometrically_plausible_plate',len(candidates))
    _,plate_index,silhouette,(x,y,bw,bh)=max(options,key=lambda item:item[0])
    plate=candidates[plate_index]['segmentation'].astype(np.uint8)
    interior=cv2.erode(silhouette.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
    rim=silhouette & ~interior
    seeds=[]
    for index,candidate in enumerate(candidates):
        mask=candidate['segmentation'].astype(bool); area=int(mask.sum())
        if index==plate_index or not 100<=area<=.65*interior.sum():
            continue
        if (mask & silhouette).sum()/area>=.85 and (mask & rim).sum()/area<=.10:
            seeds.append((str(index),mask))
    count,labels,stats,_=cv2.connectedComponentsWithStats((interior & ~plate.astype(bool)).astype(np.uint8),connectivity=8)
    for label in range(1,count):
        mask=labels==label
        if 100<=int(mask.sum())<=.65*interior.sum():
            seeds.append((f'hole_{label}',mask))
    unique=[]
    for index,mask in sorted(seeds,key=lambda item:-int(item[1].sum())):
        if any((mask & other).sum()/max(1,(mask | other).sum())>.85 for _,other in unique):
            continue
        unique.append((index,mask))
    positive=[]; sources=[]
    for index,mask in unique:
        distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
        py,px=np.unravel_index(distance.argmax(),distance.shape)
        if any(np.hypot(px-ox,py-oy)<8 for ox,oy in positive):
            continue
        positive.append([int(px),int(py)]); sources.append(index)
        if len(positive)==16:
            break
    negatives=[]
    if unique:
        union=np.logical_or.reduce([mask for _,mask in unique])
        forbidden=cv2.dilate(union.astype(np.uint8),np.ones((11,11),np.uint8)).astype(bool)
        bare=cv2.erode(plate,np.ones((9,9),np.uint8)).astype(bool)&interior&~forbidden
        bare_distance=cv2.distanceTransform(bare.astype(np.uint8),cv2.DIST_L2,5)
        yy,xx=np.indices((h,w)); cx=x+bw/2; cy=y+bh/2
        for sx,sy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
            region=bare & ((xx-cx)*sx>=0) & ((yy-cy)*sy>=0)
            value=np.where(region,bare_distance/(1+np.hypot(xx-cx,yy-cy)/50),-1)
            if value.max()>0:
                ny,nx=np.unravel_index(value.argmax(),value.shape)
                point=[int(nx),int(ny)]
                if point not in negatives:
                    negatives.append(point)
    return {'box_xyxy':[max(0,x-4),max(0,y-4),min(w-1,x+bw+4),min(h-1,y+bh+4)],
        'points_xy':positive+negatives,'point_labels':[1]*len(positive)+[0]*len(negatives),
        'positive_count':len(positive),'negative_count':len(negatives),
        'positive_seed_sources':sources,'plate_candidate_index':plate_index,
        'candidate_count':len(candidates),'seed_count':len(unique),
        'fallback_reason':None if positive else 'plate_box_only_no_structural_food_seeds',
        'reference_mask_used':False}
