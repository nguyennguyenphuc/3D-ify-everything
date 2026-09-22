"""Every test uses its own data directory; no production lock/recover/download."""
import pytest

@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    from studio import config, db, api, jobs, worker, sources, capacity
    for module in (config,db,api,jobs,worker,sources,capacity):
        monkeypatch.setattr(module,'DATA',tmp_path)
    monkeypatch.setenv('STUDIO_DATA',str(tmp_path))
    db.migrate()
    yield tmp_path
