"""In-memory stand-in for colab.ghchannel.GitHub used by agent/remote tests."""
import itertools
from colab.ghchannel import CHANNEL_TITLE, GitHubError


class FakeGitHub:
    def __init__(self, login='owner', association='OWNER'):
        self.login, self.association = login, association
        self.ids = itertools.count(1)
        self.issues, self.comments_by_issue, self.branches, self.releases = {}, {}, {}, {}

    def find_channel(self, create=False):
        for issue in self.issues.values():
            if issue['title'] == CHANNEL_TITLE: return issue
        if not create: return None
        issue = {'number': next(self.ids), 'title': CHANNEL_TITLE, 'body': ''}
        self.issues[issue['number']] = issue; self.comments_by_issue[issue['number']] = []
        return issue

    def issue(self, number): return dict(self.issues[number])
    def edit_issue(self, number, **fields): self.issues[number].update(fields); return self.issues[number]
    def comments(self, number, since=None): return [dict(c) for c in self.comments_by_issue[number]]

    def comment(self, number, body, login=None, association=None):
        c = {'id': next(self.ids), 'body': body, 'user': {'login': login or self.login},
             'author_association': association or self.association}
        self.comments_by_issue[number].append(c)
        return c

    def edit_comment(self, comment_id, body):
        for comments in self.comments_by_issue.values():
            for c in comments:
                if c['id'] == comment_id: c['body'] = body; return c
        raise GitHubError(404, 'comment')

    def publish_tree(self, branch, files, message):
        self.branches[branch] = dict(files); return f'sha-{len(self.branches)}'

    def read_file(self, branch, path):
        try: return self.branches[branch][path]
        except KeyError: raise GitHubError(404, path) from None

    def list_tree(self, branch):
        if branch not in self.branches: raise GitHubError(404, branch)
        return sorted(self.branches[branch])

    def release(self, tag, create=False, target=None, body=''):
        if tag not in self.releases:
            if not create: raise GitHubError(404, tag)
            self.releases[tag] = {'id': next(self.ids), 'tag_name': tag, 'assets': [], 'data': {}}
        return self.releases[tag]

    def upload_asset(self, release, path, name=None):
        data = open(path, 'rb').read()
        asset = {'id': next(self.ids), 'name': name, 'size': len(data)}
        release['assets'].append(asset); release['data'][asset['id']] = data
        return asset

    def download_asset(self, asset_id, dest):
        for r in self.releases.values():
            if asset_id in r['data']:
                open(dest, 'wb').write(r['data'][asset_id]); return dest
        raise GitHubError(404, 'asset')
