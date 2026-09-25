"""GitHub-backed control channel between a Colab GPU runtime and any machine.

Transport (only api.github.com / uploads.github.com are needed on both sides):
- one open issue titled ``CHANNEL_TITLE`` is the mailbox;
- the issue body holds the agent heartbeat (JSON);
- a comment starting with ``JOB_MARK`` submits a job, ``CANCEL_MARK`` cancels one;
- the agent keeps one ``STATUS_MARK`` comment per job updated with state + log tail;
- small outputs are published as a single orphan commit on ``colab-runs/<id>``;
- large outputs are release assets on tag ``colab-run-<id>``;
- private inputs are assets of the draft release ``INPUTS_RELEASE``, fetched by the agent before a job.
"""
import base64
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

API = 'https://api.github.com'
UPLOADS = 'https://uploads.github.com'
CHANNEL_TITLE = '[colab-agent] control channel'
JOB_MARK = '<!-- colab-job -->'
CANCEL_MARK = '<!-- colab-cancel -->'
STATUS_MARK = '<!-- colab-status '
HEARTBEAT_MARK = '<!-- colab-heartbeat -->'
ID_RE = re.compile(r'^[a-z0-9][a-z0-9-]{2,62}$')
TERMINAL = {'succeeded', 'failed', 'cancelled', 'rejected', 'interrupted'}
INPUTS_RELEASE = 'colab-inputs'
ASSET_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$')


class GitHubError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f'GitHub {status}: {message}')
        self.status = status


class GitHub:
    def __init__(self, repo, token=None, api=API, uploads=UPLOADS, timeout=60):
        self.repo, self.token, self.api, self.uploads, self.timeout = repo, token, api, uploads, timeout

    def request(self, method, path, body=None, *, raw=None, accept='application/vnd.github+json',
                content_type='application/json', base=None, params=None):
        url = (base or self.api) + path
        if params: url += '?' + urllib.parse.urlencode(params)
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        headers = {'Accept': accept, 'User-Agent': 'courtyard-colab-agent', 'X-GitHub-Api-Version': '2022-11-28'}
        if data is not None: headers['Content-Type'] = content_type
        if self.token: headers['Authorization'] = f'Bearer {self.token}'
        for attempt in range(5):
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = r.read()
                    if accept.endswith('json') and payload: return json.loads(payload)
                    return payload
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors='replace')[:500]
                retry = e.code in (502, 503, 504) or (e.code in (403, 429) and 'rate limit' in text.lower())
                if not retry or attempt == 4: raise GitHubError(e.code, text) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == 4: raise GitHubError(0, str(e)) from None
            time.sleep(2 ** attempt)

    def repo_path(self, suffix):
        return f'/repos/{self.repo}{suffix}'

    # Issues / comments ---------------------------------------------------
    def find_channel(self, create=False):
        for page in range(1, 6):
            issues = self.request('GET', self.repo_path('/issues'), params={'state': 'open', 'per_page': 100, 'page': page})
            for issue in issues:
                if issue.get('title') == CHANNEL_TITLE and 'pull_request' not in issue: return issue
            if len(issues) < 100: break
        if not create: return None
        body = HEARTBEAT_MARK + '\nAgent chưa chạy. Mở colab/agent_bootstrap.ipynb trên Colab.'
        return self.request('POST', self.repo_path('/issues'), {'title': CHANNEL_TITLE, 'body': body})

    def comments(self, issue, since=None):
        out, page = [], 1
        while True:
            params = {'per_page': 100, 'page': page}
            if since: params['since'] = since
            batch = self.request('GET', self.repo_path(f'/issues/{issue}/comments'), params=params)
            out += batch
            if len(batch) < 100: return out
            page += 1

    def comment(self, issue, body):
        return self.request('POST', self.repo_path(f'/issues/{issue}/comments'), {'body': body})

    def edit_comment(self, comment_id, body):
        return self.request('PATCH', self.repo_path(f'/issues/comments/{comment_id}'), {'body': body})

    def edit_issue(self, issue, **fields):
        return self.request('PATCH', self.repo_path(f'/issues/{issue}'), fields)

    def issue(self, issue):
        return self.request('GET', self.repo_path(f'/issues/{issue}'))

    # Git data: one orphan commit per publish -----------------------------
    def publish_tree(self, branch, files, message):
        """files: {path: bytes}. Replaces the branch with a single parentless commit."""
        entries = []
        for path, content in sorted(files.items()):
            blob = self.request('POST', self.repo_path('/git/blobs'),
                                {'content': base64.b64encode(content).decode(), 'encoding': 'base64'})
            entries.append({'path': path, 'mode': '100644', 'type': 'blob', 'sha': blob['sha']})
        tree = self.request('POST', self.repo_path('/git/trees'), {'tree': entries})
        commit = self.request('POST', self.repo_path('/git/commits'), {'message': message, 'tree': tree['sha'], 'parents': []})
        ref = f'heads/{branch}'
        try:
            self.request('PATCH', self.repo_path(f'/git/refs/{ref}'), {'sha': commit['sha'], 'force': True})
        except GitHubError as e:
            if e.status not in (404, 422): raise
            self.request('POST', self.repo_path('/git/refs'), {'ref': f'refs/{ref}', 'sha': commit['sha']})
        return commit['sha']

    def read_file(self, branch, path):
        return self.request('GET', self.repo_path(f'/contents/{urllib.parse.quote(path)}'), params={'ref': branch},
                            accept='application/vnd.github.raw')

    def list_tree(self, branch):
        tree = self.request('GET', self.repo_path(f'/git/trees/{urllib.parse.quote(branch, safe="")}'), params={'recursive': 1})
        return [e['path'] for e in tree.get('tree', []) if e['type'] == 'blob']

    # Releases --------------------------------------------------------------
    def release(self, tag, create=False, target=None, body=''):
        try:
            return self.request('GET', self.repo_path(f'/releases/tags/{tag}'))
        except GitHubError as e:
            if e.status != 404 or not create: raise
        payload = {'tag_name': tag, 'name': tag, 'body': body, 'prerelease': True}
        if target: payload['target_commitish'] = target
        return self.request('POST', self.repo_path('/releases'), payload)

    def draft_release(self, name, create=False):
        """A draft release is invisible to the public even on a public repo; find it by name (tags skip drafts)."""
        for page in range(1, 11):
            releases = self.request('GET', self.repo_path('/releases'), params={'per_page': 100, 'page': page})
            for r in releases:
                if r.get('draft') and r.get('name') == name: return r
            if len(releases) < 100: break
        if not create: raise GitHubError(404, f'draft release {name}')
        return self.request('POST', self.repo_path('/releases'),
                            {'tag_name': name, 'name': name, 'draft': True, 'body': 'Input riêng tư cho agent Colab.'})

    def upload_asset(self, release, path, name=None):
        name = name or os.path.basename(path)
        for asset in release.get('assets', []):
            if asset['name'] == name: self.request('DELETE', self.repo_path(f'/releases/assets/{asset["id"]}'))
        with open(path, 'rb') as f: data = f.read()
        return self.request('POST', self.repo_path(f'/releases/{release["id"]}/assets'), raw=data, base=self.uploads,
                            params={'name': name}, content_type='application/octet-stream')

    def download_asset(self, asset_id, dest):
        """Follow the storage redirect without forwarding the GitHub token."""
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs): return None
        headers = {'Accept': 'application/octet-stream', 'User-Agent': 'courtyard-colab-agent'}
        if self.token: headers['Authorization'] = f'Bearer {self.token}'
        req = urllib.request.Request(self.api + self.repo_path(f'/releases/assets/{asset_id}'), headers=headers)
        opener = urllib.request.build_opener(NoRedirect)
        try:
            response = opener.open(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            if e.code not in (301, 302, 303, 307, 308): raise GitHubError(e.code, e.read().decode(errors='replace')[:300]) from None
            response = urllib.request.urlopen(urllib.request.Request(e.headers['Location'],
                headers={'User-Agent': 'courtyard-colab-agent'}), timeout=self.timeout)
        with response, open(dest, 'wb') as f:
            while chunk := response.read(1 << 20): f.write(chunk)
        return dest


def fenced(obj):
    return '```json\n' + json.dumps(obj, ensure_ascii=False, indent=2) + '\n```'


def parse_fenced(body):
    match = re.search(r'```json\s*\n(.*?)\n```', body or '', re.S)
    if not match: return None
    try: return json.loads(match.group(1))
    except ValueError: return None


def status_marker(job_id):
    return f'{STATUS_MARK}{job_id} -->'


def runs_branch(job_id):
    return f'colab-runs/{job_id}'


def release_tag(job_id):
    return f'colab-run-{job_id}'


def new_job_id(name='job'):
    slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')[:24] or 'job'
    return f'{time.strftime("%Y%m%d-%H%M%S", time.gmtime())}-{slug}-{os.urandom(2).hex()}'
