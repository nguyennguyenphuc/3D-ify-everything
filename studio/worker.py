import json
import os
import shutil
import sys
import threading
import time
import traceback
import psutil
from . import db
from .config import DATA, ROOT, inside
from .capacity import capabilities, require_capacity


def main(id):
    job = db.job(id)
    project = DATA/'projects'/job['project_id']
    run = project/'runs'/id
    # Manager normally creates this directory. Keep the worker restartable and
    # able to finish/report a recovered job even if the manager process died.
    run.mkdir(parents=True, exist_ok=True)
    config = job['config']
    started = time.monotonic()
    origin = config.get('uploaded_at',job['created'])
    budget = min(30,config.get('budget_minutes',30))*60
    deadline = origin+budget
    report = {'job_id':id,'kind':job['kind'],'config':config,'upstream':json.loads((ROOT/'upstream.lock.json').read_text()),'stages':{},'quality_status':'requires_visual_review'}
    current = {'stage':'queued','since':time.monotonic(),'frame_count':0}
    event_lock = threading.RLock()
    expired = threading.Event()
    def save_report():
        report.update(elapsed_seconds=time.time()-origin,peak_process_tree_rss_bytes=peak[0],status=db.job(id)['status'])
        temporary=run/'report.tmp'
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        temporary.replace(run/'report.json')
    peak = [0]
    stop = threading.Event()
    def sample():
        while not stop.wait(.5):
            try:
                p = psutil.Process()
                peak[0] = max(peak[0],sum(x.memory_info().rss for x in [p]+p.children(recursive=True) if x.is_running()))
            # Sandboxed macOS may deny the process-table query used by
            # children(recursive=True); accounting must never kill a run.
            except (psutil.Error, OSError): pass
    threading.Thread(target=sample,daemon=True).start()
    def emit(message,**fields):
        with event_lock:
            stage=fields.get('stage',current['stage'])
            if stage != current['stage']:
                previous=current['stage']
                report['stages'][previous]=report['stages'].get(previous,0)+time.monotonic()-current['since']
                current.update(stage=stage,since=time.monotonic())
            current['frame_count']=fields.get('frame_count',current['frame_count'])
            event = dict(time=time.time(),message=message,elapsed_seconds=time.time()-origin,frame_count=current['frame_count'])
            event.update(fields)
            with (run/'events.jsonl').open('a') as f: f.write(json.dumps(event,ensure_ascii=False)+'\n')
            print(message,flush=True)
    db.status(id,'running')
    timed = job['kind'] in ('geometry','train','pipeline','room','video')
    def timeout():
        with event_lock:
            expired.set()
            emit('Đã hết ngân sách thời gian; giữ artifact đã ghi',stage='timeout')
            db.status(id,'timed_out','Hết ngân sách; artifact/snapshot đã ghi được giữ lại')
            report['budget_exhausted']=True
            save_report()
        import signal
        os.killpg(os.getpgrp(),signal.SIGTERM)
    timer = threading.Timer(max(.01,deadline-time.time()),timeout) if timed else None
    if timer: timer.start()
    def heartbeat():
        while not stop.wait(5):
            emit('Đang xử lý: '+current['stage'],stage=current['stage'])
    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        if job['kind'] == 'courtyard':
            from .sources import courtyard
            courtyard(project,emit)
        elif job['kind'] == 'video':
            from .sources import extract_video
            extract_video(inside(project,config['video']),project,config['fps'],emit)
        elif job['kind'] == 'import':
            from .sources import import_images
            from .config import import_path
            import_images(import_path(config['path']),project)
        else:
            if job['kind'] == 'room':
                from .sources import extract_video
                extract_video(inside(project,config['video']),project,config['fps'],emit,config.get('target_frames'))
            if job['kind'] in ('geometry','pipeline','room'):
                from .vggt_mps import reconstruct
                manifest = json.loads((project/'manifest.json').read_text())
                from .selection import select_frames
                manual=config.get('images') if config.get('selection_mode','manual')=='manual' else None
                # The fixed courtyard fixture has its own sparse-track evidence.
                if (project/'courtyard-selection.json').exists() and manual and len(manual)==8:
                    by_id={m['id']:m for m in manifest}
                    selected=[by_id[i] for i in manual]
                    selection={'images':selected,'connected':True,'method':'courtyard sparse tracks'}
                    (run/'selection.json').write_text(json.dumps(selection,indent=2))
                else:
                    selection=select_frames(project,manifest,run,emit,config.get('target_frames',32),64,manual)
                    selected=selection['images']
                report['selection']=selection
                require_capacity(len(selected))
                paths = [inside(project,m['file']) for m in selected]
                report['geometry'] = reconstruct(paths,run,emit,config['confidence'],max_points=capabilities()['max_points'])
                camera_path=run/'geometry/cameras.json'
                cameras=json.loads(camera_path.read_text())
                for camera,item in zip(cameras['cameras'],selected):
                    camera.update(source_id=item['id'],timestamp_seconds=item.get('timestamp_seconds'))
                camera_path.write_text(json.dumps(cameras,indent=2))
            else:
                parent = db.job(config['geometry_job'])
                if parent['project_id'] != job['project_id']: raise ValueError('Geometry phải thuộc cùng project')
                previous = project/'runs'/parent['id']
                for name in ('images','sparse','geometry'):
                    shutil.copytree(previous/name,run/name)
            if job['kind'] in ('train','pipeline','room'):
                from .training import train
                report['training'] = train(run,config['iterations'],deadline-time.time()-5,emit)
        exhausted=report.get('training',{}).get('budget_exhausted',False)
        db.status(id,'timed_out' if exhausted else 'succeeded', 'Hết ngân sách; giữ kết quả hợp lệ' if exhausted else None)
        emit('Hết ngân sách; đã giữ kết quả' if exhausted else 'Hoàn tất xử lý; cần kiểm tra chất lượng từ nhiều góc nhìn',stage='done',progress=1)
    except Exception as e:
        traceback.print_exc()
        message=str(e)
        if 'out of memory' in message.lower() or 'mps backend out of memory' in message.lower():
            message='MPS không đủ bộ nhớ cho số ảnh đã chọn. Không tự giảm số ảnh. '+message
        if not expired.is_set(): db.status(id,'failed',message)
        emit(message,stage='error')
        report['error'] = message
    finally:
        stop.set()
        if timer: timer.cancel()
        with event_lock:
            report['stages'][current['stage']]=report['stages'].get(current['stage'],0)+time.monotonic()-current['since']
            save_report()


if __name__ == '__main__': main(sys.argv[1])
