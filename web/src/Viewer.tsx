import {useEffect, useRef, useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls.js';
import {PLYLoader} from 'three/examples/jsm/loaders/PLYLoader.js';
import type {components} from './generated';

type CameraInfo = {name:string; timestamp_seconds?:number|null; wh:number[]; K:number[][]; viewer_c2w:number[][]; depth:string};
type Mode = 'splat'|'points'|'depth';
export default function Viewer({job, paused}:{job:components['schemas']['JobOut']; paused:boolean}) {
  const host=useRef<HTMLDivElement>(null);
  const [files,setFiles]=useState<string[]>([]), [cameras,setCameras]=useState<CameraInfo[]>([]);
  const [mode,setMode]=useState<Mode>('splat'), [view,setView]=useState('0');
  const [helpers,setHelpers]=useState(false), [size,setSize]=useState(2), [message,setMessage]=useState('Đang tải…');
  const settings=useRef({paused,helpers,size}); settings.current={paused,helpers,size};
  const base=`/api/jobs/${job.id}/artifacts/`;
  useEffect(()=>{
    const abort=new AbortController(); setFiles([]);setCameras([]);setMessage('Đang tải…');setView('0');
    (async()=>{
      const response=await fetch(base.slice(0,-1),{signal:abort.signal}); if(!response.ok)throw Error('Không tải được danh sách kết quả');
      const paths=(await response.json() as {path:string}[]).map(f=>f.path);
      let cs:CameraInfo[]=[];
      if(paths.includes('geometry/cameras.json')) {
        const r=await fetch(base+'geometry/cameras.json',{signal:abort.signal}); if(!r.ok)throw Error('Không tải được camera'); cs=(await r.json()).cameras;
      }
      if(abort.signal.aborted)return;
      setFiles(paths);setCameras(cs);setMode(paths.includes('training/splat.ply')?'splat':'points');
      setMessage(paths.includes('geometry/points.ply')||paths.includes('training/splat.ply')?'Đang tải cảnh 3D…':'Chưa có kết quả 3D. Viewer sẽ cập nhật khi job kết thúc.');
    })().catch(e=>{if(!abort.signal.aborted)setMessage(String(e))});
    return()=>abort.abort();
  },[base,job.status]);

  useEffect(()=>{
    const el=host.current;
    const path=mode==='splat'?'training/splat.ply':'geometry/points.ply';
    if(!el||mode==='depth'||!files.includes(path))return;
    let disposed=false, frame=0;
    const abort=new AbortController(); const releases:(()=>void)[]=[];
    setMessage('Đang tải cảnh 3D…');
    (async()=>{try {
      const scene=new THREE.Scene();scene.background=new THREE.Color('#101312');
      const camera=new THREE.PerspectiveCamera(55,1,.001,10000);
      const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));
      el.appendChild(renderer.domElement);releases.push(()=>{renderer.dispose();renderer.domElement.remove()});
      const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;releases.push(()=>controls.dispose());
      const response=await fetch(base+'geometry/points.ply',{signal:abort.signal});
      let bounds=new THREE.Sphere(new THREE.Vector3(),2);
      if(response.ok){
        const geometry=new PLYLoader().parse(await response.arrayBuffer());
        if(disposed){geometry.dispose();return}
        releases.push(()=>geometry.dispose());geometry.computeBoundingSphere();bounds=geometry.boundingSphere!;
        if(mode==='points') {
          const material=new THREE.PointsMaterial({size:settings.current.size,vertexColors:true,sizeAttenuation:false});
          releases.push(()=>material.dispose());scene.add(new THREE.Points(geometry,material));
        }
      }
      if(disposed)return;
      const radius=Math.max(bounds.radius,.01);
      camera.near=Math.max(radius/10000,.00001);camera.far=radius*100+100;
      const source=view==='all'?undefined:cameras[Number(view)];
      if(source){
        // JSON stores rows; Three.js fromArray consumes columns.
        const pose=new THREE.Matrix4().fromArray(source.viewer_c2w.flat()).transpose();
        pose.decompose(camera.position,camera.quaternion,camera.scale);
        const forward=new THREE.Vector3(0,0,-1).applyQuaternion(camera.quaternion);
        const distance=Math.max(bounds.center.clone().sub(camera.position).dot(forward),radius*.2);
        controls.target.copy(camera.position).addScaledVector(forward,distance);
        controls.object.up.set(0,1,0).applyQuaternion(camera.quaternion);
      }else{controls.target.copy(bounds.center);camera.position.copy(bounds.center).add(new THREE.Vector3(1,.6,1).multiplyScalar(radius*1.6));}
      const cameraGroup=new THREE.Group();scene.add(cameraGroup);
      for(const c of cameras){
        const helperCamera=new THREE.PerspectiveCamera(2*Math.atan(c.wh[1]/(2*c.K[1][1]))*180/Math.PI,c.wh[0]/c.wh[1],radius*.002,radius*.08);
        new THREE.Matrix4().fromArray(c.viewer_c2w.flat()).transpose().decompose(helperCamera.position,helperCamera.quaternion,helperCamera.scale);
        helperCamera.updateMatrixWorld(true);
        const helper=new THREE.CameraHelper(helperCamera);cameraGroup.add(helper);releases.push(()=>helper.dispose());
      }
      let draw=()=>renderer.render(scene,camera);
      if(mode==='splat'){
        const {SplatMesh,SparkRenderer}=await import('@sparkjsdev/spark');if(disposed)return;
        const mesh=new SplatMesh({url:base+path});releases.push(()=>mesh.dispose());
        await mesh.initialized;if(disposed)return;
        scene.add(mesh);const spark=new SparkRenderer({renderer});scene.add(spark);releases.push(()=>spark.dispose());
        draw=()=>spark.render(scene,camera);
      }
      const resize=()=>{
        const w=el.clientWidth,h=el.clientHeight;renderer.setSize(w,h,false);
        camera.aspect=w/h;camera.updateProjectionMatrix();
        if(source){
          // Letterbox the calibrated source camera, retaining principal point.
          const [iw,ih]=source.wh, k=source.K, n=camera.near, f=camera.far;
          const scale=Math.min(w/iw,h/ih), vw=iw*scale,vh=ih*scale;
          renderer.setViewport((w-vw)/2,(h-vh)/2,vw,vh);
          camera.projectionMatrix.makePerspective(-k[0][2]*n/k[0][0],(iw-k[0][2])*n/k[0][0],k[1][2]*n/k[1][1],-(ih-k[1][2])*n/k[1][1],n,f);
          camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
        }
      };
      const observer=new ResizeObserver(resize);observer.observe(el);releases.push(()=>observer.disconnect());resize();
      const render=()=>{if(disposed)return;frame=requestAnimationFrame(render);if(document.hidden||settings.current.paused)return;
        controls.update();cameraGroup.visible=settings.current.helpers;
        scene.traverse(o=>{if(o instanceof THREE.Points)(o.material as THREE.PointsMaterial).size=settings.current.size});draw();
      };render();setMessage(mode==='splat'?'Gaussian Splatting · kéo để xoay, cuộn để phóng to':'Point cloud · hình học thưa, depth có thang đo tương đối');
    }catch(e){if(!disposed)setMessage(`Lỗi hiển thị ${mode}: ${e instanceof Error?e.message:String(e)}`)}})();
    return()=>{disposed=true;abort.abort();cancelAnimationFrame(frame);releases.reverse().forEach(release=>release())};
  },[base,files,cameras,mode,view]);
  const source=cameras[view==='all'?0:Number(view)];
  return <section className="result-viewer">
    <div className="viewer-tools" aria-label="Chế độ xem">
      <button aria-pressed={mode==='splat'} disabled={!files.includes('training/splat.ply')} onClick={()=>setMode('splat')}>Splat 3D</button>
      <button aria-pressed={mode==='points'} disabled={!files.includes('geometry/points.ply')} onClick={()=>setMode('points')}>Point cloud</button>
      <button aria-pressed={mode==='depth'} disabled={!cameras.length} onClick={()=>setMode('depth')}>Depth</button>
      <select aria-label="Góc nhìn" value={view} onChange={e=>setView(e.target.value)}><option value="all">Toàn cảnh</option>{cameras.map((c,i)=><option key={c.name} value={String(i)}>Camera {i+1} · {c.timestamp_seconds != null ? `${c.timestamp_seconds.toFixed(1)}s` : c.name}</option>)}</select>
      {mode!=='depth'&&<label><input type="checkbox" checked={helpers} onChange={e=>setHelpers(e.target.checked)}/>Hiện camera</label>}
      {mode==='points'&&<label>Cỡ điểm<input aria-label="Cỡ điểm" type="range" min="1" max="6" step=".5" value={size} onChange={e=>setSize(Number(e.target.value))}/></label>}
    </div>
    <div className="viewer">{mode==='depth'&&source?<div className="depth-pair"><figure><img src={base+'images/'+source.name}/><figcaption>Ảnh nguồn</figcaption></figure><figure><img src={base+source.depth}/><figcaption>Depth tương đối · không phải mét</figcaption></figure></div>:<div ref={host} className="canvas"/>}<p role="status">{paused?'Tạm dừng viewer trong lúc GPU xử lý.':mode==='depth'?'Depth theo ảnh nguồn':message}</p></div>
    {files.includes('training/splat.ply')&&<a href={base+'training/splat.ply'} download>Tải cảnh Gaussian Splatting (.ply)</a>}
  </section>;
}
