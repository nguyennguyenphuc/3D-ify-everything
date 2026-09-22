from fastapi.testclient import TestClient
from studio.api import app


def test_health_and_local_security():
    with TestClient(app) as client:
        assert client.get('/api/health').json() == {'ok': True}
        response=client.get('/api/health',headers={'Origin':'https://example.invalid'})
        assert response.status_code == 403


def test_manual_geometry_requires_multiple_images():
    with TestClient(app) as client:
        p=client.post('/api/projects',json={'name':'API validation fixture'}).json()
        response=client.post(f'/api/projects/{p["id"]}/jobs',json={'kind':'geometry','images':['0'],'selection_mode':'manual'})
        assert response.status_code == 422


def test_video_creates_new_project_when_source_is_occupied(monkeypatch):
    from studio.api import DATA
    with TestClient(app) as client:
        original=client.post('/api/projects',json={'name':'has images'}).json()
        project_dir=DATA/'projects'/original['id']
        (project_dir/'manifest.json').write_text('[]')
        started=[]
        def start(project_id,kind,config):
            started.append((project_id,kind,config))
            return {'id':'job','project_id':project_id,'kind':kind,'status':'queued','config':config,'created':0,'updated':0,'error':None}
        monkeypatch.setattr('studio.api.manager.start',start)
        response=client.post(f'/api/projects/{original["id"]}/video?create_project_if_occupied=true',files={'file':('walk.mp4',b'video','video/mp4')})
        assert response.status_code == 202
        created=response.json()
        assert created['project_id'] != original['id']
        assert started[0][0] == created['project_id']
        assert started[0][1] == 'room'
        assert started[0][2]['target_frames'] == 32
        assert started[0][2]['budget_minutes'] == 30
        assert started[0][2]['uploaded_at'] > 0
        assert (project_dir/'manifest.json').exists()


def test_source_run_can_restart_without_manager_directory(monkeypatch):
    monkeypatch.setattr('studio.sources.courtyard',lambda project,emit: [])
    # Worker creates its own run directory as well, so a recovered job can
    # persist events/report after an API-manager restart.
    from studio import db
    from studio.worker import main
    project=db.create_project('worker recovery fixture')
    job=db.new_job(project['id'],'courtyard',{})
    main(job['id'])
    assert db.job(job['id'])['status'] == 'succeeded'


def test_capacity_gate_and_large_manual_validation(monkeypatch):
    from studio.api import JobIn
    assert len(JobIn(kind='geometry',selection_mode='manual',images=[str(i) for i in range(32)]).images)==32
    with TestClient(app) as client:
        project=client.post('/api/projects',json={'name':'capacity'}).json()
        response=client.post(f"/api/projects/{project['id']}/jobs",json={'kind':'pipeline','target_frames':32})
        assert response.status_code==409
        assert 'MPS' in response.json()['detail']
        assert client.get('/api/capabilities').json()['max_frames']==8


def test_extract_only_video(monkeypatch):
    def start(project_id,kind,config):
        return {'id':'test','project_id':project_id,'kind':kind,'config':config,'created':0,'updated':0,'status':'queued','error':None}
    monkeypatch.setattr('studio.api.manager.start',start)
    with TestClient(app) as client:
        p=client.post('/api/projects',json={'name':'extract'}).json()
        result=client.post(f"/api/projects/{p['id']}/video?reconstruct=false",files={'file':('v.mp4',b'video','video/mp4')})
        assert result.status_code==202
        assert result.json()['kind']=='video'
