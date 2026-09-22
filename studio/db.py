import json
import sqlite3
import time
import uuid
from .config import DATA, AppError


def connect():
    DATA.mkdir(parents=True,exist_ok=True)
    db = sqlite3.connect(DATA/'studio.sqlite',timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    return db


def migrate():
    with connect() as db:
        db.executescript('''CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY,name TEXT,created REAL);
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,project_id TEXT,kind TEXT,status TEXT,config TEXT,created REAL,updated REAL,error TEXT);
        PRAGMA user_version=1;''')


def recover():
    with connect() as db:
        db.execute("UPDATE jobs SET status='interrupted',error='Ứng dụng đã dừng trước khi job hoàn tất',updated=? WHERE status IN ('running','queued','cancelling')",(time.time(),))


def projects():
    with connect() as db: return [dict(r) for r in db.execute('SELECT * FROM projects ORDER BY created DESC')]


def create_project(name):
    row = {'id':uuid.uuid4().hex,'name':name,'created':time.time()}
    with connect() as db: db.execute('INSERT INTO projects VALUES (:id,:name,:created)',row)
    (DATA/'projects'/row['id']).mkdir(parents=True)
    return row


def project(id):
    with connect() as db: row = db.execute('SELECT * FROM projects WHERE id=?',(id,)).fetchone()
    if row is None: raise AppError('Không tìm thấy project',404)
    return dict(row)


def job(id):
    with connect() as db: row = db.execute('SELECT * FROM jobs WHERE id=?',(id,)).fetchone()
    if row is None: raise AppError('Không tìm thấy job',404)
    result = dict(row); result['config'] = json.loads(result['config'])
    return result


def jobs(project_id):
    with connect() as db: ids = [r[0] for r in db.execute('SELECT id FROM jobs WHERE project_id=? ORDER BY created DESC',(project_id,))]
    return [job(id) for id in ids]


def new_job(project_id,kind,config):
    id = uuid.uuid4().hex
    now = time.time()
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        if db.execute("SELECT 1 FROM jobs WHERE status IN ('running','queued','cancelling')").fetchone():
            raise AppError('Đã có job đang chạy. Hãy chờ hoặc hủy job đó.',409)
        db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)',(id,project_id,kind,'queued',json.dumps(config),now,now,None))
    return job(id)


def status(id,value,error=None):
    with connect() as db: db.execute('UPDATE jobs SET status=?,error=?,updated=? WHERE id=?',(value,error,time.time(),id))
