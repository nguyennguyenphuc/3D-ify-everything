"""Deterministic temporal selection; overlap is evidence, not a quality certificate."""
import json
import cv2
import numpy as np
from .config import AppError, inside


class Selector:
    def __init__(self, project, manifest, emit):
        self.manifest = manifest
        self.emit = emit
        self.features = []
        self.pairs = {}
        self.rejected = []
        detector = cv2.ORB_create(nfeatures=1800)
        hashes = []
        for i, item in enumerate(manifest):
            image = cv2.imread(str(inside(project,item['file'])), cv2.IMREAD_GRAYSCALE)
            if image is None: raise AppError(f"Không đọc được ảnh {item['id']}")
            h,w = image.shape
            image = cv2.resize(image,(max(1,round(w*640/max(h,w))),max(1,round(h*640/max(h,w)))))
            sharpness = float(cv2.Laplacian(image,cv2.CV_64F).var())
            small = cv2.resize(image,(32,32)).astype(np.float32)
            dct = cv2.dct(small)[:8,:8].flatten()[1:]
            signature = dct > np.median(dct)
            # Compare all accepted candidates: returning to a location should not
            # spend the image budget on near-identical views.
            duplicate = any(np.count_nonzero(np.logical_xor(signature, old)) <= 3 and abs(float(small.mean()) - mean) < 8 for old, mean in hashes)
            reason = 'blur' if sharpness < 25 else 'duplicate' if duplicate else None
            keypoints, descriptors = detector.detectAndCompute(image,None)
            self.features.append((np.array([k.pt for k in keypoints],np.float32),descriptors,sharpness))
            if reason: self.rejected.append({'id':item['id'],'reason':reason,'sharpness':sharpness})
            else: hashes.append((signature,float(small.mean())))
            if i % 50 == 0: emit(f'Đánh giá ảnh {i+1}/{len(manifest)}',stage='selection',frame_count=i+1)
        rejected = {r['id'] for r in self.rejected}
        self.eligible = [i for i,m in enumerate(manifest) if m['id'] not in rejected]

    def overlap(self, a, b):
        key = tuple(sorted((a,b)))
        if key in self.pairs: return self.pairs[key]
        pa,da,_ = self.features[a]; pb,db,_ = self.features[b]
        inliers = 0
        if da is not None and db is not None and min(len(da),len(db)) >= 12:
            matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2)
            good = [m[0] for m in matches if len(m)==2 and m[0].distance < .75*m[1].distance]
            if len(good)>=12:
                _, mask = cv2.findFundamentalMat(np.array([pa[m.queryIdx] for m in good]),
                    np.array([pb[m.trainIdx] for m in good]),cv2.FM_RANSAC,2.,.99)
                if mask is not None: inliers = int(mask.sum())
        self.pairs[key] = inliers
        return inliers

    def gaps(self, selected):
        return [{'left':self.manifest[a]['id'],'right':self.manifest[b]['id'],
            'start_seconds':self.manifest[a].get('timestamp_seconds'),
            'end_seconds':self.manifest[b].get('timestamp_seconds'),
            'inliers':self.overlap(a,b)} for a,b in zip(selected,selected[1:]) if self.overlap(a,b)<12]

    def select(self, target, maximum, manual=None):
        if manual is not None:
            by_id = {m['id']:i for i,m in enumerate(self.manifest)}
            selected = sorted(by_id[i] for i in manual)
        else:
            selected=[]
            # Temporal bins include the entire video, including the last bin.
            for bucket in np.array_split(np.arange(len(self.manifest)),min(target,len(self.manifest))):
                choices = [int(i) for i in bucket if i in self.eligible]
                if choices: selected.append(max(choices,key=lambda i:self.features[i][2]))
            for i in sorted(self.eligible,key=lambda i:-self.features[i][2]):
                if len(selected)>=target: break
                if i not in selected: selected.append(i)
            selected.sort()
        attempts=[]
        for level in sorted(set([len(selected)]+[n for n in (48,64) if len(selected)<n<=maximum])):
            if manual is None:
                while len(selected)<level:
                    broken = [(a,b) for a,b in zip(selected,selected[1:]) if self.overlap(a,b)<12]
                    additions=[]
                    for a,b in broken:
                        choices=[i for i in self.eligible if a<i<b and i not in selected]
                        if choices:
                            # Prefer a connecting frame; temporal midpoint otherwise.
                            mid=(a+b)/2
                            choices=sorted(choices,key=lambda i:abs(i-mid))[:9]
                            additions.append(max(choices,key=lambda i:(min(self.overlap(a,i),self.overlap(i,b))>=12,-abs(i-mid),self.features[i][2])))
                    if not additions: break
                    selected=sorted(set(selected+additions[:level-len(selected)]))
            gaps=self.gaps(selected)
            attempts.append({'limit':level,'count':len(selected),'gaps':gaps})
            if len(selected)>=2 and not gaps: break
        gaps=self.gaps(selected)
        return {'images':[self.manifest[i] for i in selected], 'requested_count':target,
            'selected_count':len(selected),'candidate_count':len(self.manifest),
            'rejected':self.rejected,'rejected_count':len(self.rejected),
            'unselected_count':len(self.manifest)-len(selected), 'attempts':attempts,
            'gaps':gaps,'connected':len(selected)>=2 and not gaps,
            'method':'temporal bins, Laplacian sharpness, perceptual duplicates, ORB/RANSAC adjacent overlap',
            'quality_status':'requires_visual_review'}


def select_frames(project, manifest, run, emit, target=32, maximum=64, manual=None):
    result=Selector(project,manifest,emit).select(target,maximum,manual)
    (run/'selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    emit(f"Đã chọn {result['selected_count']} ảnh; loại {result['rejected_count']} ảnh mờ/gần trùng",
         stage='selection',frame_count=result['selected_count'],gaps=result['gaps'])
    if not result['connected']:
        spans=', '.join(f"{g['start_seconds']}–{g['end_seconds']} giây" if g['start_seconds'] is not None else f"{g['left']}–{g['right']}" for g in result['gaps'][:8])
        raise AppError('Thiếu liên kết giữa các góc nhìn. Quay bổ sung chậm, giữ vùng chung tại: '+(spans or 'video thiếu ảnh rõ khác nhau'))
    return result
