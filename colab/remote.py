"""Drive the Colab agent from any machine that can reach api.github.com.

    python -m colab.remote status
    python -m colab.remote run "nvidia-smi"                 # submit, stream log, exit with job result
    python -m colab.remote submit "python -m colab.pipeline run --source courtyard" --name courtyard
    python -m colab.remote logs <id> --follow
    python -m colab.remote files <id>
    python -m colab.remote fetch <id> publish/metrics.json [dest]
    python -m colab.remote fetch <id> splat.ply [dest]      # release asset
    python -m colab.remote cancel <id>

Auth: GH_TOKEN/GITHUB_TOKEN if set; otherwise no header (an authenticating proxy may add one).
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from .ghchannel import (GitHub, GitHubError, CANCEL_MARK, JOB_MARK, TERMINAL, fenced, new_job_id, parse_fenced,
                        release_tag, runs_branch, status_marker)


def default_repo():
    if os.environ.get('COLAB_REPO'): return os.environ['COLAB_REPO']
    try:
        url = subprocess.run(['git', 'remote', 'get-url', 'origin'], capture_output=True, text=True).stdout.strip()
        match = re.search(r'github\.com[:/]([^/]+/[^/.]+?)(?:\.git)?$', url)
        if match: return match.group(1)
    except OSError:
        pass
    return 'Thanhjash/3D-ify-everything'


class Remote:
    def __init__(self, gh, out=sys.stdout, sleep=time.sleep):
        self.gh, self.out, self.sleep = gh, out, sleep
        self._issue = None

    def say(self, text=''):
        print(text, file=self.out, flush=True)

    def issue(self, create=False):
        if self._issue is None:
            self._issue = self.gh.find_channel(create=create)
            if self._issue is None: raise SystemExit('Chưa có kênh điều khiển. Chạy colab/agent_bootstrap.ipynb trên Colab trước.')
        return self._issue

    def heartbeat(self):
        issue = self.gh.issue(self.issue()['number'])
        data = parse_fenced(issue.get('body')) or {}
        if data.get('time'):
            age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.strptime(data['time'], '%Y-%m-%dT%H:%M:%SZ')
                   .replace(tzinfo=dt.timezone.utc)).total_seconds()
            data['age_seconds'] = round(age)
        return data

    def status(self):
        data = self.heartbeat()
        if not data.get('time'):
            self.say(f'Issue #{self.issue()["number"]}: agent chưa gửi heartbeat. Chạy notebook bootstrap trên Colab.')
            return 1
        gpu = data.get('gpu') or {}
        alive = data['age_seconds'] < 180
        self.say(f'Agent {"ĐANG CHẠY" if alive else "MẤT KẾT NỐI"} · heartbeat {data["age_seconds"]} giây trước · issue #{self.issue()["number"]}')
        self.say(f'GPU: {gpu.get("name", "không có")} · VRAM {gpu.get("memory_used_mib", "?")}/{gpu.get("memory_total_mib", "?")} MiB')
        self.say(f'Code: {data.get("code_branch")} @ {str(data.get("code_commit"))[:10]} · job đang chạy: {data.get("busy_job")} · hàng đợi: {data.get("queued")}')
        return 0 if alive else 2

    def submit(self, cmd, name='job', timeout_minutes=240, pull=True, env=None, branch=None):
        job_id = new_job_id(name)
        spec = {'id': job_id, 'cmd': cmd, 'timeout_minutes': timeout_minutes, 'pull': pull, 'env': env or {}}
        if branch: spec['branch'] = branch
        self.gh.comment(self.issue(create=True)['number'], JOB_MARK + '\n' + fenced(spec))
        self.say(f'Đã gửi job {job_id}')
        return job_id

    def job_status(self, job_id):
        marker = status_marker(job_id)
        for c in reversed(self.gh.comments(self.issue()['number'])):
            body = c.get('body') or ''
            if body.startswith(marker):
                data = parse_fenced(body) or {}
                match = re.search(r'```text\n(.*)\n```\s*$', body, re.S)
                data['_tail'] = match.group(1).split('\n') if match and match.group(1) else []
                return data
        return None

    def logs(self, job_id, follow=False, interval=10, timeout=None):
        printed, start = 0, time.time()
        while True:
            data = self.job_status(job_id)
            if data:
                total, tail_start, tail = data.get('log_total_lines', 0), data.get('tail_start', 0), data['_tail']
                if printed < tail_start:
                    self.say(f'… bỏ qua {tail_start - printed} dòng (đầy đủ ở log.txt trên nhánh {runs_branch(job_id)})')
                    printed = tail_start
                for line in tail[printed - tail_start:total - tail_start]:
                    self.say(line)
                printed = max(printed, total)
                if data.get('state') in TERMINAL:
                    self.say(f'== {job_id}: {data["state"]} · exit {data.get("exit_code")} · {data.get("elapsed_seconds")} giây'
                             + (f' · {data["error"]}' if data.get('error') else ''))
                    return data
            elif not follow:
                self.say('Job chưa được agent nhận.')
                return None
            if not follow: return data
            if timeout and time.time() - start > timeout: raise SystemExit(f'Hết thời gian chờ job {job_id}')
            self.sleep(interval)

    def files(self, job_id):
        try:
            for path in self.gh.list_tree(runs_branch(job_id)): self.say(f'branch  {path}')
        except GitHubError as e:
            self.say(f'Chưa có nhánh {runs_branch(job_id)} ({e.status})')
        try:
            for asset in self.gh.release(release_tag(job_id)).get('assets', []):
                self.say(f'release {asset["name"]}  {asset["size"]/2**20:.1f} MiB')
        except GitHubError:
            pass

    def fetch(self, job_id, name, dest=None):
        dest = Path(dest or Path(name).name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            for asset in self.gh.release(release_tag(job_id)).get('assets', []):
                if asset['name'] == name:
                    self.gh.download_asset(asset['id'], dest); self.say(f'Đã tải {dest}'); return dest
        except GitHubError as e:
            if e.status != 404: raise
        dest.write_bytes(self.gh.read_file(runs_branch(job_id), name))
        self.say(f'Đã tải {dest}')
        return dest

    def cancel(self, job_id):
        self.gh.comment(self.issue()['number'], CANCEL_MARK + '\n' + fenced({'id': job_id}))
        self.say(f'Đã yêu cầu hủy {job_id}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', default=default_repo())
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('status')
    for name in ('submit', 'run'):
        p = sub.add_parser(name)
        p.add_argument('cmd')
        p.add_argument('--name', default='job')
        p.add_argument('--timeout', type=float, default=240, help='phút')
        p.add_argument('--no-pull', action='store_true')
        p.add_argument('--branch')
        p.add_argument('--env', action='append', default=[], help='KEY=VALUE')
    p = sub.add_parser('logs'); p.add_argument('id'); p.add_argument('--follow', action='store_true')
    p = sub.add_parser('wait'); p.add_argument('id'); p.add_argument('--timeout', type=float, default=None, help='phút')
    p = sub.add_parser('files'); p.add_argument('id')
    p = sub.add_parser('fetch'); p.add_argument('id'); p.add_argument('name'); p.add_argument('dest', nargs='?')
    p = sub.add_parser('cancel'); p.add_argument('id')
    args = parser.parse_args(argv)
    remote = Remote(GitHub(args.repo, os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')))
    if args.command == 'status': return remote.status()
    if args.command in ('submit', 'run'):
        env = dict(x.split('=', 1) for x in args.env)
        job_id = remote.submit(args.cmd, args.name, args.timeout, not args.no_pull, env, args.branch)
        if args.command == 'submit': return 0
        data = remote.logs(job_id, follow=True, timeout=(args.timeout + 30) * 60)
        return 0 if data and data.get('state') == 'succeeded' else 1
    if args.command == 'logs': remote.logs(args.id, follow=args.follow); return 0
    if args.command == 'wait':
        data = remote.logs(args.id, follow=True, timeout=args.timeout * 60 if args.timeout else None)
        return 0 if data.get('state') == 'succeeded' else 1
    if args.command == 'files': remote.files(args.id); return 0
    if args.command == 'fetch': remote.fetch(args.id, args.name, args.dest); return 0
    if args.command == 'cancel': remote.cancel(args.id); return 0


if __name__ == '__main__':
    sys.exit(main())
