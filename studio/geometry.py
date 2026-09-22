"""OpenCV world→camera conventions. All pixel transforms are explicit affine maps."""
import json
import struct
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement


def preprocess(path: Path, size=518, train_max=1024):
    with Image.open(path) as source:
        img = ImageOps.exif_transpose(source).convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (size - rw) // 2, (size - rh) // 2
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(img.resize((rw, rh), Image.Resampling.BICUBIC), (left, top))
    mask = np.zeros((size, size), bool)
    mask[top:top + rh, left:left + rw] = True
    train_scale = min(1, train_max / max(w, h))
    tw, th = round(w * train_scale), round(h * train_scale)
    # Pixel centers: u' = s * (u + 0.5) - 0.5 + padding.
    A = np.array([[rw/w, 0, left + (rw/w - 1)/2], [0, rh/h, top + (rh/h - 1)/2], [0, 0, 1]])
    B = np.array([[tw/w, 0, (tw/w - 1)/2], [0, th/h, (th/h - 1)/2], [0, 0, 1]])
    meta = {"original_wh": [w,h], "resized_wh": [rw,rh], "padding_lt": [left,top], "train_wh": [tw,th], "original_to_model": A.tolist(), "original_to_train": B.tolist()}
    return np.asarray(canvas), mask, img.resize((tw, th), Image.Resampling.LANCZOS), meta


def train_intrinsics(K, meta):
    return np.asarray(meta["original_to_train"]) @ np.linalg.inv(meta["original_to_model"]) @ K


def unproject(depth, K, w2c):
    yy, xx = np.indices(depth.shape)
    uv = np.stack([xx, yy, np.ones_like(xx)], -1)
    camera = (uv @ np.linalg.inv(K).T) * depth[..., None]
    return (camera - w2c[:3, 3]) @ w2c[:3, :3]


def project(xyz, K, w2c):
    camera = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    pixels = camera @ K.T
    return pixels[..., :2] / pixels[..., 2:], camera[..., 2]


def camera_to_viewer(w2c):
    T = np.eye(4)
    T[:3] = w2c[:3]
    return np.linalg.inv(T) @ np.diag([1, -1, -1, 1])


def write_ply(path, points, colors):
    vertices = np.empty(len(points), dtype=[(n, 'f4') for n in ('x','y','z')] + [(n,'u1') for n in ('red','green','blue')])
    for i,n in enumerate(('x','y','z')): vertices[n] = points[:,i]
    for i,n in enumerate(('red','green','blue')): vertices[n] = colors[:,i]
    PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(str(path))


def write_colmap(root, cameras, points, colors, observations):
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    # One observation per retained depth point; reciprocal image tracks are valid.
    tracks = [[] for _ in cameras]
    for pid, (image_idx, u, v) in enumerate(observations, 1):
        tracks[int(image_idx)].append((u, v, pid))
    with (sparse / "cameras.bin").open('wb') as f:
        f.write(struct.pack('<Q', len(cameras)))
        for i,c in enumerate(cameras, 1):
            K = np.asarray(c['K'])
            f.write(struct.pack('<iiQQdddd', i, 1, *c['wh'], K[0,0], K[1,1], K[0,2], K[1,2]))
    with (sparse / "images.bin").open('wb') as f:
        f.write(struct.pack('<Q', len(cameras)))
        for i,c in enumerate(cameras, 1):
            E = np.asarray(c['w2c'])
            q = Rotation.from_matrix(E[:3,:3]).as_quat()[[3,0,1,2]]
            f.write(struct.pack('<i4d3di', i, *q, *E[:3,3], i))
            f.write(c['name'].encode() + b'\0')
            f.write(struct.pack('<Q', len(tracks[i-1])))
            for u,v,pid in tracks[i-1]: f.write(struct.pack('<ddq',u,v,pid))
    track_indices = [0] * len(cameras)
    with (sparse / "points3D.bin").open('wb') as f:
        f.write(struct.pack('<Q',len(points)))
        for pid,(xyz,rgb,obs) in enumerate(zip(points,colors,observations),1):
            idx = int(obs[0])
            f.write(struct.pack('<QdddBBBdQii',pid,*xyz,*rgb,0.,1,idx+1,track_indices[idx]))
            track_indices[idx] += 1


def consistent_points(points, source, depths, confidences, masks, extrinsics, intrinsics, quantile=.5):
    """Require agreement with at least one nearby view within 5% relative depth."""
    supported=np.zeros(len(points),bool)
    neighbors=[j for j in range(max(0,source-2),min(len(depths),source+3)) if j!=source]
    for j in neighbors:
        uv,z=project(points,intrinsics[j],extrinsics[j])
        finite=np.isfinite(uv).all(1)&np.isfinite(z)&(z>0)
        xy=np.rint(np.where(np.isfinite(uv),uv,-1)).astype(np.int64)
        h,w=depths[j].shape
        valid=finite&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        ids=np.flatnonzero(valid)
        x,y=xy[ids,0],xy[ids,1]
        d=depths[j][y,x]; c=confidences[j][y,x]
        pool=masks[j]&np.isfinite(depths[j])&(depths[j]>0)&np.isfinite(confidences[j])
        if not pool.any(): continue
        threshold=np.quantile(confidences[j][pool],quantile)
        supported[ids] |= masks[j][y,x]&np.isfinite(d)&(d>0)&np.isfinite(c)&(c>=threshold)&(np.abs(d-z[ids])<=.05*np.maximum(d,z[ids]))
    return supported


def export(root, arrays, masks, metas, names, depth, conf, extrinsics, intrinsics, quantile=.5, max_points=100000):
    output = root / 'geometry'
    output.mkdir(exist_ok=True)
    cameras, points, colors, observations = [], [], [], []
    rng = np.random.default_rng(42)
    consistency=[]
    for i,(d,c,E,K) in enumerate(zip(depth,conf,extrinsics,intrinsics)):
        if not np.isfinite(E).all() or not np.isfinite(K).all() or min(K[0,0],K[1,1]) <= 0:
            raise ValueError(f'Camera {i} không hợp lệ')
        if not np.allclose(E[:3,:3] @ E[:3,:3].T, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(E[:3,:3]),1,atol=1e-3):
            raise ValueError(f'Camera {i}: rotation không hợp lệ')
        valid = masks[i] & np.isfinite(d) & (d > 0) & np.isfinite(c)
        if not valid.any(): raise ValueError(f'Ảnh {i} không có depth hữu hạn dương')
        threshold = float(np.quantile(c[valid],quantile))
        valid &= c >= threshold
        world = unproject(d,K,E)
        valid &= np.isfinite(world).all(-1)
        yy,xx = np.where(valid)
        if len(xx) == 0: raise ValueError(f'Ảnh {i} không có điểm sau lọc')
        keep = rng.choice(len(xx), min(len(xx),4*max_points//len(names)),replace=False)
        yy,xx = yy[keep],xx[keep]
        pts = world[yy,xx]
        supported=consistent_points(pts,i,depth,conf,masks,extrinsics,intrinsics,quantile)
        consistency.append(float(supported.mean()))
        yy,xx=yy[supported],xx[supported]
        if len(xx)<8: raise ValueError(f'Ảnh {i}: quá ít điểm nhất quán giữa các góc nhìn; không thể xác nhận cảnh liên kết')
        keep=rng.choice(len(xx),min(len(xx),max_points//len(names)),replace=False)
        yy,xx=yy[keep],xx[keep]
        pts=world[yy,xx]
        uv,_ = project(pts,train_intrinsics(K,metas[i]),E)
        points.append(pts); colors.append(arrays[i][yy,xx]); observations.append(np.column_stack([np.full(len(xx),i),uv]))
        np.savez_compressed(output/f'depth_{i:02}.npz',depth=d,confidence=c,valid=valid)
        display = np.zeros((*d.shape,3),np.uint8)
        low,high = np.percentile(d[valid],[2,98])
        shade = np.clip((np.nan_to_num(d,nan=low)-low)/max(high-low,1e-8),0,1)
        display[...,0] = (shade*220).astype('uint8'); display[...,1] = ((1-shade)*180).astype('uint8'); display[...,2] = 120
        display[~valid] = 0
        Image.fromarray(display).save(output/f'depth_{i:02}.png')
        cameras.append({'name':names[i], 'wh':metas[i]['train_wh'], 'K':train_intrinsics(K,metas[i]).tolist(), 'model_K':K.tolist(), 'w2c':E.tolist(),'viewer_c2w':camera_to_viewer(E).tolist(),'transform':metas[i], 'depth':f'geometry/depth_{i:02}.png','valid_depth_fraction':float(valid.mean()),'confidence_threshold':threshold})
    points,colors,observations = np.concatenate(points),np.concatenate(colors),np.concatenate(observations)
    write_ply(output/'points.ply',points,colors)
    write_colmap(root,cameras,points,colors,observations)
    (output/'cameras.json').write_text(json.dumps({'convention':'OpenCV world-to-camera; viewer c2w = inverse(w2c) @ diag(1,-1,-1,1)', 'depth_scale':'relative', 'cameras':cameras},indent=2))
    return {'camera_count':len(cameras),'point_count':len(points),'depth_scale':'relative','finite_depth':bool(np.isfinite(depth).all()),'cross_view_supported_fraction':consistency}
