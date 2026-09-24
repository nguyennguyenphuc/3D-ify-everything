"""Colab-side agent: executes jobs posted to the GitHub control issue.

Run inside Colab (see colab/agent_bootstrap.ipynb):
    GH_TOKEN=... python -m colab.agent --repo OWNER/REPO --workdir /content/3D-ify-everything

Only comments from the repo owner/members/collaborators (or ``--allow`` logins)
are executed. Anyone who can write such a comment can run shell commands on
this Colab runtime, so only add trusted collaborators and keep the token repo-scoped.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from .ghchannel import (GitHub, GitHubError, CANCEL_MARK, HEARTBEAT_MARK, ID_RE, JOB_MARK, TERMINAL, fenced,
                        parse_fenced, release_tag, runs_branch, status_marker)

VERSION = 1
TRUSTED = {'OWNER', 'MEMBER', 'COLLABORATOR'}
TAIL_LINES = 80
PUBLISH_LIMIT = 50 * 1024 * 1024
FILE_LIMIT = 45 * 1024 * 1024
ASSET_LIMIT = 1900 * 1024 * 1024


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def gpu_info():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10).stdout
        name, total, used, util = [x.strip() for x in out.strip().splitlines()[0].split(',')]
        return {'name': name, 'memory_total_mib': int(total), 'memory_used_mib': int(used), 'utilization': int(util)}
    except Exception:
        return None


def git(workdir, *args):
    return subprocess.run(['git', '-C', str(workdir), *args], capture_output=True, text=True, timeout=300)


class Job:
    def __init__(self, spec, comment, root):
        self.spec, self.comment = spec, comment
        self.id = spec['id']
        self.dir = root/self.id
        self.log_path = self.dir/'log.txt'
        self.publish_dir = self.dir/'publish'
        self.release_dir = self.dir/'release'
        self.process = None
        self.started = None
        self.finished = None
        self.exit_code = None
        self.state = 'queued'
        self.status_comment = None
        self.commit = None
        self.error = None
        self.cancel_requested = False
        self.last_status = 0
        self.last_publish = 0
        self.published_commit = None
        self.assets = []

    @property
    def timeout(self):
        return float(self.spec.get('timeout_minutes', 240)) * 60


class Agent:
    def __init__(self, gh, workdir, root, branch, allow=(), poll=10, status_every=20, publish_every=60,
                 heartbeat_every=60, clock=time.time):
        self.gh, self.workdir, self.root, self.branch = gh, Path(workdir), Path(root), branch
        self.allow, self.poll = set(allow), poll
        self.status_every, self.publish_every, self.heartbeat_every = status_every, publish_every, heartbeat_every
        self.clock = clock
        self.root.mkdir(parents=True, exist_ok=True)
        self.issue = None
        self.queue, self.done, self.current = [], set(), None
        self.rejected = set()
        self.last_heartbeat = 0
        self.started = now()

    def log(self, message):
        print(f'[agent {time.strftime("%H:%M:%S")}] {message}', flush=True)

    # Discovery ----------------------------------------------------------
    def trusted(self, comment):
        login = (comment.get('user') or {}).get('login', '')
        return comment.get('author_association') in TRUSTED or login in self.allow

    def scan(self):
        comments = self.gh.comments(self.issue['number'])
        statuses, cancels = {}, set()
        for c in comments:
            body = c.get('body') or ''
            data = parse_fenced(body)
            if body.startswith('<!-- colab-status ') and data and data.get('id'):
                statuses[data['id']] = (c, data)
            elif body.startswith(CANCEL_MARK) and data and self.trusted(c):
                cancels.add(str(data.get('id')))
        for c in comments:
            body = c.get('body') or ''
            if not body.startswith(JOB_MARK): continue
            spec = parse_fenced(body) or {}
            job_id = str(spec.get('id', ''))
            if job_id in self.done or (self.current and self.current.id == job_id) or any(j.id == job_id for j in self.queue):
                continue
            if job_id in statuses:
                comment, data = statuses[job_id]
                if data.get('state') in TERMINAL:
                    self.done.add(job_id); continue
                # Status exists but this agent does not own it: the previous agent died.
                self.done.add(job_id)
                data.update(state='interrupted', error='Agent Colab đã khởi động lại giữa chừng; gửi lại job.', finished=now())
                self.gh.edit_comment(comment['id'], status_marker(job_id) + '\n' + fenced(data))
                continue
            if not self.trusted(c):
                if c['id'] not in self.rejected:
                    self.rejected.add(c['id'])
                    self.log(f'Bỏ qua job từ {c["user"]["login"]} ({c.get("author_association")}); thêm --allow nếu tin cậy')
                continue
            problem = self.validate(spec)
            job = Job(spec if not problem else {'id': job_id or f'invalid-{c["id"]}'}, c, self.root)
            if problem:
                job.state, job.error, job.finished = 'rejected', problem, now()
                self.done.add(job.id)
                self.post_status(job)
                continue
            if job_id in cancels:
                job.state, job.error, job.finished = 'cancelled', 'Đã hủy trước khi chạy', now()
                self.done.add(job.id); self.post_status(job); continue
            self.queue.append(job)
            self.log(f'Nhận job {job.id}: {spec["cmd"][:120]}')
        if self.current and self.current.id in cancels and not self.current.cancel_requested:
            self.cancel(self.current)

    @staticmethod
    def validate(spec):
        if not isinstance(spec, dict): return 'Job phải là JSON object'
        if not ID_RE.match(str(spec.get('id', ''))): return 'id không hợp lệ'
        if not isinstance(spec.get('cmd'), str) or not spec['cmd'].strip(): return 'Thiếu cmd'
        env = spec.get('env', {})
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            return 'env phải là {str: str}'
        try:
            if not 0 < float(spec.get('timeout_minutes', 240)) <= 24 * 60: return 'timeout_minutes ngoài khoảng'
        except (TypeError, ValueError):
            return 'timeout_minutes không hợp lệ'
        return None

    # Execution ----------------------------------------------------------
    def start(self, job):
        for d in (job.dir, job.publish_dir, job.release_dir): d.mkdir(parents=True, exist_ok=True)
        with job.log_path.open('w') as log:
            if job.spec.get('pull', True) and (self.workdir/'.git').exists():
                branch = job.spec.get('branch', self.branch)
                for args in (('fetch', '--quiet', 'origin', branch), ('reset', '--hard', f'origin/{branch}')):
                    r = git(self.workdir, *args)
                    log.write(f'$ git {" ".join(args)}\n{r.stdout}{r.stderr}')
                    if r.returncode != 0:
                        job.state, job.error, job.exit_code = 'failed', f'git {args[0]} thất bại', r.returncode
                        job.started = job.finished = now()
                        return False
            job.commit = (git(self.workdir, 'rev-parse', 'HEAD').stdout.strip() or None) if (self.workdir/'.git').exists() else None
            log.write(f'$ {job.spec["cmd"]}\n')
        env = dict(os.environ, PYTHONUNBUFFERED='1', COLAB_JOB_ID=job.id, COLAB_JOB_DIR=str(job.dir),
                   COLAB_PUBLISH_DIR=str(job.publish_dir), COLAB_RELEASE_DIR=str(job.release_dir))
        for secret in ('GH_TOKEN', 'GITHUB_TOKEN'): env.pop(secret, None)
        env.update(job.spec.get('env', {}))
        log = job.log_path.open('a')
        job.process = subprocess.Popen(['bash', '-lc', job.spec['cmd']], cwd=self.workdir, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True, env=env)
        job._log_handle = log
        job.state, job.started, job._t0 = 'running', now(), self.clock()
        return True

    def cancel(self, job):
        job.cancel_requested = True
        job._kill_at = self.clock() + 30
        if job.process and job.process.poll() is None:
            self.log(f'Hủy job {job.id}')
            try: os.killpg(job.process.pid, signal.SIGTERM)
            except ProcessLookupError: pass

    def tick(self, job):
        """Advance a running job; returns True when it has finished."""
        code = job.process.poll()
        elapsed = self.clock() - job._t0
        if code is None and elapsed > job.timeout and not job.cancel_requested:
            job.error = f'Quá thời gian {job.timeout/60:.0f} phút'
            self.cancel(job)
        if code is None and job.cancel_requested and self.clock() > job._kill_at:
            try: os.killpg(job.process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        if code is None: return False
        job._log_handle.close()
        job.exit_code, job.finished = code, now()
        if job.cancel_requested and not job.error: job.state = 'cancelled'
        elif job.error: job.state = 'failed'
        else: job.state = 'succeeded' if code == 0 else 'failed'
        return True

    # Reporting ----------------------------------------------------------
    def tail(self, job):
        try: lines = job.log_path.read_text(errors='replace').splitlines()
        except OSError: lines = []
        tail = lines[-TAIL_LINES:]
        text = '\n'.join(tail)
        while len(text) > 50000 and tail:
            tail = tail[1:]; text = '\n'.join(tail)
        return len(lines), len(lines) - len(tail), text

    def status_payload(self, job):
        total, start, _ = self.tail(job)
        elapsed = (self.clock() - job._t0) if hasattr(job, '_t0') else 0
        return {'id': job.id, 'state': job.state, 'cmd': job.spec.get('cmd'), 'commit': job.commit,
                'started': job.started, 'finished': job.finished, 'elapsed_seconds': round(elapsed),
                'exit_code': job.exit_code, 'error': job.error, 'gpu': gpu_info(),
                'log_total_lines': total, 'tail_start': start, 'branch': runs_branch(job.id),
                'published_commit': job.published_commit, 'release': release_tag(job.id) if job.assets else None,
                'assets': job.assets, 'updated': now()}

    def post_status(self, job):
        _, _, text = self.tail(job)
        body = status_marker(job.id) + '\n' + fenced(self.status_payload(job)) + '\n\n```text\n' + text.replace('```', "'''") + '\n```'
        if job.status_comment: self.gh.edit_comment(job.status_comment, body)
        else: job.status_comment = self.gh.comment(self.issue['number'], body)['id']
        job.last_status = self.clock()

    def collect(self, job):
        files = {'log.txt': job.log_path.read_bytes() if job.log_path.exists() else b''}
        files['status.json'] = json.dumps(self.status_payload(job), ensure_ascii=False, indent=2).encode()
        total = sum(len(v) for v in files.values())
        for path in sorted(job.publish_dir.rglob('*')):
            if not path.is_file() or path.is_symlink(): continue
            size = path.stat().st_size
            if size > FILE_LIMIT or total + size > PUBLISH_LIMIT: continue
            files['publish/' + path.relative_to(job.publish_dir).as_posix()] = path.read_bytes()
            total += size
        return files

    def publish(self, job):
        job.published_commit = self.gh.publish_tree(runs_branch(job.id), self.collect(job), f'colab run {job.id}: {job.state}')
        job.last_publish = self.clock()

    def upload_release(self, job):
        files = [p for p in sorted(job.release_dir.rglob('*')) if p.is_file() and not p.is_symlink() and p.stat().st_size <= ASSET_LIMIT]
        if not files: return
        release = self.gh.release(release_tag(job.id), create=True, target=None,
                                  body=f'Kết quả Colab job `{job.id}` (commit {job.commit}).')
        for path in files:
            name = path.relative_to(job.release_dir).as_posix().replace('/', '__')
            asset = self.gh.upload_asset(release, str(path), name)
            job.assets.append({'name': asset['name'], 'id': asset['id'], 'size': asset['size']})

    def heartbeat(self, force=False):
        if not force and self.clock() - self.last_heartbeat < self.heartbeat_every: return
        head = git(self.workdir, 'rev-parse', 'HEAD').stdout.strip() if (self.workdir/'.git').exists() else None
        data = {'time': now(), 'agent_version': VERSION, 'agent_started': self.started, 'gpu': gpu_info(),
                'busy_job': self.current.id if self.current else None, 'queued': [j.id for j in self.queue],
                'code_branch': self.branch, 'code_commit': head, 'python': sys.version.split()[0],
                'disk_free_gb': round(shutil.disk_usage(self.root).free / 2**30, 1)}
        body = HEARTBEAT_MARK + '\nAgent Colab đang chạy. Gửi job bằng `python -m colab.remote submit "<lệnh>"`.\n\n' + fenced(data)
        self.gh.edit_issue(self.issue['number'], body=body)
        self.last_heartbeat = self.clock()

    def finish(self, job):
        for step in (self.upload_release, self.publish, self.post_status):
            try: step(job)
            except Exception as e:
                job.error = (job.error + '; ' if job.error else '') + f'{step.__name__}: {e}'
                self.log(f'{step.__name__} lỗi: {e}')
        self.done.add(job.id)
        self.log(f'Job {job.id} kết thúc: {job.state} (exit {job.exit_code})')

    def step(self):
        if self.issue is None:
            self.issue = self.gh.find_channel(create=True)
            self.log(f'Kênh điều khiển: issue #{self.issue["number"]}')
            self.heartbeat(force=True)
        self.scan()
        if self.current is None and self.queue:
            job = self.queue.pop(0)
            if self.start(job):
                self.current = job
                self.post_status(job)
            else:
                self.finish(job)
        if self.current:
            job = self.current
            if self.tick(job):
                self.current = None
                self.finish(job)
            else:
                if self.clock() - job.last_status >= self.status_every: self.post_status(job)
                if self.clock() - job.last_publish >= self.publish_every: self.publish(job)
        self.heartbeat()

    def run(self, once=False):
        while True:
            try:
                self.step()
            except GitHubError as e:
                # A rejected token never recovers by retrying; stop with a fix instead of looping.
                if e.status == 401:
                    if self.current: self.cancel(self.current)
                    raise SystemExit('GitHub 401 Bad credentials: GH_TOKEN sai, hết hạn hoặc đã bị xoá. Tạo lại token '
                                     f'fine-grained cho {self.gh.repo}, dán vào Colab Secrets → GH_TOKEN, chạy lại ô 1 và ô 2.') from None
                self.log(f'GitHub tạm lỗi: {e}')
            except KeyboardInterrupt:
                if self.current: self.cancel(self.current)
                raise
            if once and not self.current and not self.queue: return
            time.sleep(self.poll)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default=os.environ.get('COLAB_REPO', 'nguyennguyenphuc/3D-ify-everything'))
    parser.add_argument('--workdir', default=os.getcwd())
    parser.add_argument('--root', default='/content/colab_jobs')
    parser.add_argument('--branch', default=os.environ.get('COLAB_CODE_BRANCH', 'feature/courtyard-studio'))
    parser.add_argument('--allow', default=os.environ.get('COLAB_AGENT_ALLOW', ''), help='comma-separated extra logins')
    parser.add_argument('--poll', type=float, default=10)
    parser.add_argument('--once', action='store_true', help='exit when no job is running or queued')
    args = parser.parse_args(argv)
    token = (os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN') or '').strip()
    if not token: raise SystemExit('Thiếu GH_TOKEN (Colab Secrets)')
    agent = Agent(GitHub(args.repo, token), args.workdir, args.root, args.branch,
                  allow=[x for x in args.allow.split(',') if x], poll=args.poll)
    agent.run(once=args.once)


if __name__ == '__main__':
    main()
