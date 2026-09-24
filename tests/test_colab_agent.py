import io
import time
from colab.agent import Agent
from colab.ghchannel import JOB_MARK, fenced, parse_fenced, runs_branch, release_tag, status_marker
from colab.remote import Remote
from tests.fake_github import FakeGitHub


def make(tmp_path, gh=None):
    gh = gh or FakeGitHub()
    agent = Agent(gh, tmp_path/'repo', tmp_path/'jobs', 'feature/x', poll=0, status_every=0, publish_every=0, heartbeat_every=0)
    (tmp_path/'repo').mkdir(exist_ok=True)
    return gh, agent, Remote(gh, out=io.StringIO(), sleep=lambda s: None)


def drive(agent, job_id, limit=400):
    for _ in range(limit):
        agent.step()
        if job_id in agent.done: return
        time.sleep(.02)
    raise AssertionError('job did not finish')


def status(gh, job_id):
    issue = gh.find_channel()
    for c in gh.comments(issue['number']):
        if c['body'].startswith(status_marker(job_id)): return parse_fenced(c['body'])


def test_job_runs_publishes_and_uploads(tmp_path):
    gh, agent, remote = make(tmp_path)
    job = remote.submit('echo hello; echo "$COLAB_JOB_ID" > "$COLAB_PUBLISH_DIR/id.txt"; head -c 2048 /dev/zero > "$COLAB_RELEASE_DIR/big.bin"; echo "token=${GH_TOKEN:-none}"',
                        name='Smoke Test', env={'EXTRA': '1'})
    drive(agent, job)
    data = status(gh, job)
    assert data['state'] == 'succeeded' and data['exit_code'] == 0
    files = gh.branches[runs_branch(job)]
    assert b'hello' in files['log.txt'] and b'token=none' in files['log.txt']
    assert files['publish/id.txt'].decode().strip() == job
    assert [a['name'] for a in gh.releases[release_tag(job)]['assets']] == ['big.bin']
    heartbeat = parse_fenced(gh.find_channel()['body'])
    assert heartbeat['agent_version'] == 1 and heartbeat['busy_job'] is None
    out = remote.logs(job)
    assert out['state'] == 'succeeded' and 'hello' in remote.out.getvalue()
    remote.fetch(job, 'big.bin', tmp_path/'big.bin')
    remote.fetch(job, 'publish/id.txt', tmp_path/'id.txt')
    assert (tmp_path/'big.bin').stat().st_size == 2048 and (tmp_path/'id.txt').read_text().strip() == job


def test_failed_exit_code_is_reported(tmp_path):
    gh, agent, remote = make(tmp_path)
    job = remote.submit('echo boom; exit 3')
    drive(agent, job)
    data = status(gh, job)
    assert data['state'] == 'failed' and data['exit_code'] == 3


def test_untrusted_and_invalid_jobs_do_not_run(tmp_path):
    gh, agent, remote = make(tmp_path)
    issue = gh.find_channel(create=True)
    marker = tmp_path/'ran'
    gh.comment(issue['number'], JOB_MARK + '\n' + fenced({'id': 'abc-untrusted', 'cmd': f'touch {marker}'}), login='stranger', association='NONE')
    gh.comment(issue['number'], JOB_MARK + '\n' + fenced({'id': 'abc-invalid'}))
    for _ in range(5): agent.step()
    assert not marker.exists()
    assert status(gh, 'abc-untrusted') is None
    assert status(gh, 'abc-invalid')['state'] == 'rejected'


def test_allow_list_trusts_bot_login(tmp_path):
    gh = FakeGitHub(login='claude[bot]', association='NONE')
    _, agent, remote = make(tmp_path, gh)
    agent.allow = {'claude[bot]'}
    job = remote.submit('echo ok')
    drive(agent, job)
    assert status(gh, job)['state'] == 'succeeded'


def test_cancel_running_job(tmp_path):
    gh, agent, remote = make(tmp_path)
    job = remote.submit('sleep 30')
    for _ in range(3): agent.step()
    assert agent.current and agent.current.id == job
    remote.cancel(job)
    start = time.time()
    drive(agent, job)
    assert status(gh, job)['state'] == 'cancelled' and time.time() - start < 10


def test_restarted_agent_marks_orphaned_job_interrupted(tmp_path):
    gh, agent, remote = make(tmp_path)
    job = remote.submit('echo later')
    issue = gh.find_channel()
    gh.comment(issue['number'], status_marker(job) + '\n' + fenced({'id': job, 'state': 'running'}))
    agent.step()
    assert status(gh, job)['state'] == 'interrupted'
    assert job in agent.done and agent.current is None


def test_logs_follow_prints_each_line_once(tmp_path):
    gh, agent, remote = make(tmp_path)
    job = remote.submit('for i in 1 2 3; do echo line-$i; done')
    drive(agent, job)
    remote.logs(job, follow=True)
    text = remote.out.getvalue()
    assert all(text.count(f'line-{i}') == 1 for i in (1, 2, 3))
