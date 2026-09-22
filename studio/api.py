import asyncio
from contextlib import asynccontextmanager
import fcntl
import json
import logging
import time
import uuid
from typing import Literal
from pathlib import Path
from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, model_validator
from . import db
from .config import DATA, ROOT, AppError, inside, import_path
from .jobs import manager
from .capacity import capabilities, require_capacity


class ProjectIn(BaseModel):
    name: str = Field(min_length=1,max_length=100)


class ProjectOut(ProjectIn):
    id: str
    created: float


class JobIn(BaseModel):
    kind: Literal['courtyard','import','geometry','train','pipeline']
    images: list[str] = Field(default_factory=list,max_length=64)
    selection_mode: Literal['auto','manual'] = 'auto'
    target_frames: int = Field(default=32,ge=2,le=64)
    path: str | None = None
    confidence: float = Field(default=.5,ge=0,le=.99)
    iterations: int = Field(default=10000,ge=100,le=10000)
    budget_minutes: int = Field(default=30,ge=1,le=30)
    geometry_job: str | None = None

    @model_validator(mode='after')
    def validate_mode(self):
        if self.kind in ('geometry','pipeline') and self.selection_mode == 'manual' and (len(self.images)<2 or len(set(self.images))!=len(self.images)):
            raise ValueError('Chọn ít nhất 2 ảnh khác nhau')
        if self.kind == 'import' and not self.path: raise ValueError('Thiếu thư mục ảnh')
        if self.kind == 'train' and not self.geometry_job: raise ValueError('Thiếu geometry job')
        return self


class JobOut(BaseModel):
    id: str
    project_id: str
    kind: str
    status: str
    config: dict
    created: float
    updated: float
    error: str | None


class ImageOut(BaseModel):
    id: str
    name: str
    file: str
    thumbnail: str
    sha256: str
    timestamp_seconds: float | None = None


class Artifact(BaseModel):
    path: str
    bytes: int


@asynccontextmanager
async def lifespan(app):
    DATA.mkdir(parents=True,exist_ok=True)
    lock = (DATA/'server.lock').open('w')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Một Studio server khác đang dùng cùng data directory')
    db.migrate(); db.recover()
    yield
    manager.close()
    lock.close()


app = FastAPI(title='Courtyard Studio',version='0.1.0',lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','testserver'])
logger = logging.getLogger('studio')


@app.middleware('http')
async def local_security(request: Request,call_next):
    request_id = uuid.uuid4().hex[:12]
    origin = request.headers.get('origin')
    allowed = {'http://127.0.0.1:8000','http://localhost:8000','http://127.0.0.1:5173','http://localhost:5173'}
    if (origin and origin not in allowed) or request.headers.get('sec-fetch-site') == 'cross-site':
        return JSONResponse({'detail':'Chỉ cho phép truy cập từ ứng dụng local'},status_code=403)
    start = time.monotonic()
    try: response = await call_next(request)
    except Exception:
        logger.exception('request failed %s',request_id)
        response = JSONResponse({'detail':'Lỗi server; xem console log','request_id':request_id},status_code=500)
    response.headers.update({'X-Request-ID':request_id,'X-Content-Type-Options':'nosniff','X-Frame-Options':'DENY','Referrer-Policy':'same-origin'})
    logger.info(json.dumps({'request_id':request_id,'method':request.method,'path':request.url.path,'status':response.status_code,'seconds':time.monotonic()-start}))
    return response


@app.exception_handler(AppError)
async def app_error(request,e): return JSONResponse({'detail':e.message},status_code=e.status)


@app.get('/api/health')
def health(): return {'ok':True}


@app.get('/api/environment')
def environment():
    from .environment import inspect
    return inspect(probe=True)


@app.get('/api/capabilities')
def capacity(): return capabilities()


@app.get('/api/ready')
def ready():
    from .environment import inspect
    result = inspect()
    return JSONResponse(result,status_code=200 if result['ready'] else 503)


@app.get('/api/projects',response_model=list[ProjectOut])
def projects(): return db.projects()


@app.post('/api/projects',response_model=ProjectOut,status_code=201)
def create_project(body:ProjectIn): return db.create_project(body.name)


@app.get('/api/projects/{id}/images',response_model=list[ImageOut])
def images(id:str):
    db.project(id)
    manifest = DATA/'projects'/id/'manifest.json'
    return json.loads(manifest.read_text()) if manifest.exists() else []


@app.get('/api/projects/{id}/files/{path:path}')
def source_file(id:str,path:str):
    db.project(id)
    if Path(path).parts[0] not in ('thumbnails','originals'): raise AppError('Không được truy cập file này',403)
    p = inside(DATA/'projects'/id,path)
    if not p.is_file(): raise AppError('Không tìm thấy file',404)
    return FileResponse(p)


@app.post('/api/projects/{id}/jobs',response_model=JobOut,status_code=202)
def start_job(id:str,body:JobIn):
    db.project(id)
    config = body.model_dump(exclude={'kind'})
    project = DATA/'projects'/id
    if body.kind == 'import': import_path(body.path)
    if body.kind in ('import','courtyard') and (project/'manifest.json').exists():
        raise AppError('Project đã có nguồn ảnh. Tạo project mới để giữ dữ liệu gốc.',409)
    if body.kind in ('geometry','pipeline'):
        require_capacity(len(body.images) if body.selection_mode == 'manual' else body.target_frames)
        available = {i['id'] for i in images(id)}
        if not set(body.images) <= available: raise AppError('Ảnh đã chọn không tồn tại')
    if body.kind == 'train':
        parent = db.job(body.geometry_job)
        if parent['project_id'] != id or not (project/'runs'/parent['id']/'geometry'/'metrics.json').exists():
            raise AppError('Chưa có geometry hoàn tất trong project này')
    return manager.start(id,body.kind,config)


@app.post('/api/projects/{id}/video',response_model=JobOut,status_code=202)
async def video(id:str,file:UploadFile,fps:float=1,create_project_if_occupied:bool=False,reconstruct:bool=True,target_frames:int=32):
    db.project(id)
    if not 2 <= target_frames <= 64: raise AppError('Số ảnh mục tiêu phải từ 2 đến 64')
    if not .1 <= fps <= 10: raise AppError('FPS phải từ 0.1 đến 10')
    project = DATA/'projects'/id
    if (project/'manifest.json').exists():
        if not create_project_if_occupied:
            raise AppError('Tạo project mới để nhập video khác',409)
        stem = Path(file.filename or 'video').stem.strip() or 'video'
        created = db.create_project(f'Video · {stem[:90]}')
        id = created['id']
        project = DATA/'projects'/id
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in ('.mp4','.mov','.mkv','.avi','.webm'): raise AppError('Định dạng video không hỗ trợ')
    path = project/f'video-{uuid.uuid4().hex}{suffix}'
    total = 0
    try:
        with path.open('wb') as f:
            while chunk := await file.read(1024*1024):
                total += len(chunk)
                if total > 4*1024**3: raise AppError('Video vượt quá 4 GiB',413)
                f.write(chunk)
        return manager.start(id,'room' if reconstruct else 'video',{'video':path.name,'fps':fps,
            'selection_mode':'auto','target_frames':target_frames,'confidence':.5,
            'iterations':10000,'budget_minutes':30,'uploaded_at':time.time()})
    except Exception:
        path.unlink(missing_ok=True); raise


@app.get('/api/projects/{id}/jobs',response_model=list[JobOut])
def jobs(id:str): db.project(id); return db.jobs(id)


@app.get('/api/jobs/{id}',response_model=JobOut)
def job(id:str): return db.job(id)


@app.post('/api/jobs/{id}/cancel',response_model=JobOut)
def cancel(id:str): return manager.cancel(id)


def run_path(id):
    row = db.job(id)
    return DATA/'projects'/row['project_id']/'runs'/id


@app.get('/api/jobs/{id}/artifacts',response_model=list[Artifact])
def artifacts(id:str):
    run = run_path(id)
    return [{'path':str(p.relative_to(run)),'bytes':p.stat().st_size} for p in sorted(run.rglob('*')) if p.is_file() and p.resolve().is_relative_to(run)]


@app.get('/api/jobs/{id}/artifacts/{path:path}')
def artifact(id:str,path:str):
    p = inside(run_path(id),path)
    if not p.is_file(): raise AppError('Artifact chưa có',404)
    return FileResponse(p)


@app.get('/api/jobs/{id}/events')
async def events(id:str,request:Request):
    run = run_path(id)
    try: after = max(0,int(request.headers.get('last-event-id','0')))
    except ValueError: after = 0
    async def stream():
        cursor = 0
        count = 0
        while not await request.is_disconnected():
            path = run/'events.jsonl'
            if path.exists():
                with path.open() as f:
                    f.seek(cursor)
                    for line in f:
                        count += 1
                        if count > after: yield f'id: {count}\ndata: {line.strip()}\n\n'
                    cursor = f.tell()
            row = db.job(id)
            if row['status'] not in ('queued','running','cancelling'):
                yield f'event: done\ndata: {json.dumps(row)}\n\n'; break
            yield ': heartbeat\n\n'
            await asyncio.sleep(.75)
    return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})


if (ROOT/'web/dist').exists(): app.mount('/',StaticFiles(directory=ROOT/'web/dist',html=True),name='web')
