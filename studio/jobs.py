import json
import os
import signal
import subprocess
import sys
import threading
import time
from . import db
from .config import DATA, ROOT, AppError


class Manager:
    def __init__(self):
        self.lock = threading.RLock()
        self.processes = {}

    def start(self,project_id,kind,config):
        with self.lock:
            row = db.new_job(project_id,kind,config)
            run = DATA/'projects'/project_id/'runs'/row['id']
            run.mkdir(parents=True)
            (run/'config.json').write_text(json.dumps(row,ensure_ascii=False,indent=2))
            env = dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0',PYTHONUNBUFFERED='1')
            try:
                log = (run/'console.log').open('w')
                p = subprocess.Popen([sys.executable,'-m','studio.worker',row['id']],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                self.processes[row['id']] = p
                threading.Thread(target=self.watch,args=(row['id'],p,log),daemon=True).start()
            except Exception as e:
                db.status(row['id'],'failed',str(e)); raise
            return row

    def watch(self,id,p,log):
        p.wait(); log.close()
        with self.lock:
            row = db.job(id)
            if row['status'] == 'cancelling': db.status(id,'cancelled','Đã hủy bởi người dùng')
            elif row['status'] in ('queued','running'):
                db.status(id,'failed',f'Worker kết thúc bất thường (exit {p.returncode}); xem console.log')
            self.processes.pop(id,None)

    def cancel(self,id):
        with self.lock:
            row = db.job(id)
            if row['status'] not in ('running','queued','cancelling'): return row
            p = self.processes.get(id)
            if p is None: raise AppError('Worker không thuộc phiên hiện tại',409)
            db.status(id,'cancelling')
            try: os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            def force():
                try: p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try: os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
            threading.Thread(target=force,daemon=True).start()
            return db.job(id)

    def close(self):
        for id in list(self.processes): self.cancel(id)
        for p in list(self.processes.values()):
            try: p.wait(timeout=7)
            except subprocess.TimeoutExpired: pass


manager = Manager()
