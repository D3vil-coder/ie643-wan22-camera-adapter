#!/usr/bin/env python
"""Drive a remote marimo (Molab) kernel from the local machine: run code, upload files.

Credentials come from env vars MOLAB_URL / MOLAB_TOKEN (or --url / --token).
Never hardcode or commit them.

  python tools/molab_sync.py exec -c "print(1+1)"
  python tools/molab_sync.py exec script.py          # or '-' for stdin
  python tools/molab_sync.py push LOCAL REMOTE_PATH   # chunked, resumable, sha256-verified
  python tools/molab_sync.py pushdir LOCAL_DIR REMOTE_DIR [--glob '*.png']
  python tools/molab_sync.py pull REMOTE LOCAL        # chunked, resumable, sha256-verified
  python tools/molab_sync.py pulldir REMOTE_DIR LOCAL_DIR [--glob '*.png']
  python tools/molab_sync.py sync-code [--root /home/marimo/work/IE643-Novelty]
      # incremental git bundle of commits not yet on Molab, pushed + ff-merged there

Code runs in marimo's scratchpad (same protocol as the marimo-pair skill's
execute-code.sh): top-level bindings are discarded after each call; stdout is returned.
"""
import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

CHUNK = 768 * 1024  # raw bytes per request (~1 MB base64)


class Molab:
    """Client for one remote marimo (Molab) kernel: ``exec`` runs code in the notebook scratchpad (only stdout comes back), ``push`` /
    ``pull`` move files in ~768 KB chunks with sha256 verification and resume, ``sync_code`` ships unmerged git commits as an incremental
    bundle and fast-forwards the remote checkout.
    """

    def __init__(self, url, token):
        self.base = url.rstrip("/")
        self.headers = {"User-Agent": "curl/8.0", **({"Authorization": f"Bearer {token}"} if token else {})}
        self.session = None

    def _session(self):
        if self.session:
            return self.session
        req = urllib.request.Request(self.base + "/api/sessions", headers=self.headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            sessions = json.load(r)
        if len(sessions) != 1:
            raise SystemExit(f"expected exactly 1 session, got {list(sessions)}")
        self.session = next(iter(sessions))
        return self.session

    def exec(self, code, timeout=600, echo=True):
        """Run code; returns (ok, stdout, stderr)."""
        body = json.dumps({"code": code}).encode()
        h = {"Content-Type": "application/json", "Marimo-Session-Id": self._session(), **self.headers}
        req = urllib.request.Request(self.base + "/api/kernel/execute", data=body, headers=h, method="POST")
        out, err, ok, event, done = [], [], True, None, False
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    p = json.loads(line[5:].strip())
                    if event == "stdout":
                        out.append(p["data"])
                    elif event == "stderr":
                        err.append(p["data"])
                    elif event == "done":
                        ok = p.get("success", True) is not False
                        o = (p.get("output") or {}).get("data")
                        if o:
                            out.append(str(o))
                        done = True
                        break
        if not done:
            raise RuntimeError("stream ended without done event")
        so, se = "".join(out), "".join(err)
        if echo:
            sys.stdout.write(so)
            sys.stderr.write(se)
        return ok, so, se

    def push(self, local, remote, verbose=True):
        """Upload a local file to a remote path in chunks; resumable and sha256-verified."""
        size = os.path.getsize(local)
        h = hashlib.sha256()
        with open(local, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        digest = h.hexdigest()
        ok, so, _ = self.exec(
            f"import os;p=os.path.expanduser({remote!r});os.makedirs(os.path.dirname(p) or '.',exist_ok=True)\n"
            "print(os.path.getsize(p) if os.path.exists(p) else 0)", echo=False)
        if not ok:
            raise RuntimeError("remote path prep failed")
        off = int(so.strip().splitlines()[-1])
        if off > size:
            off = 0
        if off:
            # resume only if remote prefix is intact; else restart
            ok, so, _ = self.exec(
                "import hashlib,os\np=os.path.expanduser(%r)\nprint(hashlib.sha256(open(p,'rb').read()).hexdigest())" % remote, echo=False)
            with open(local, "rb") as f:
                ph = hashlib.sha256(f.read(off)).hexdigest()
            if so.strip() != ph:
                off = 0
        with open(local, "rb") as f:
            f.seek(off)
            first = off == 0
            while True:
                data = f.read(CHUNK)
                if not data and not first:
                    break
                b64 = base64.b64encode(data).decode()
                mode = "wb" if first else "ab"
                code = (f"import base64,os\np=os.path.expanduser({remote!r})\n"
                        f"open(p,{mode!r}).write(base64.b64decode({b64!r}))\nprint(os.path.getsize(p))")
                ok, so, se = self.exec(code, echo=False)
                if not ok:
                    raise RuntimeError("chunk failed: " + se)
                first = False
                off += len(data)
                if verbose:
                    print(f"\r{remote}: {off}/{size}", end="", flush=True)
                if not data:
                    break
        ok, so, _ = self.exec(
            "import hashlib,os\np=os.path.expanduser(%r)\nprint(hashlib.sha256(open(p,'rb').read()).hexdigest())" % remote, echo=False)
        rd = so.strip().splitlines()[-1]
        if verbose:
            print()
        if rd != digest:
            raise SystemExit(f"SHA256 MISMATCH {remote}: local {digest} remote {rd}")
        if verbose:
            print(f"OK {remote} sha256={digest}")
        return digest

    def _remote_sha(self, remote):
        ok, so, se = self.exec(
            "import hashlib,os\np=os.path.expanduser(%r)\nh=hashlib.sha256()\n"
            "f=open(p,'rb')\nfor b in iter(lambda:f.read(1<<20),b''): h.update(b)\nprint(h.hexdigest())" % remote, echo=False)
        if not ok:
            raise RuntimeError("remote sha failed: " + se)
        return so.strip().splitlines()[-1]

    def pull(self, remote, local, verbose=True):
        """Download a remote file to a local path in chunks; resumable and sha256-verified."""
        ok, so, se = self.exec("import os\np=os.path.expanduser(%r)\nprint(os.path.getsize(p))" % remote, echo=False)
        if not ok:
            raise RuntimeError(f"remote stat failed for {remote}: {se}")
        size = int(so.strip().splitlines()[-1])
        digest = self._remote_sha(remote)
        os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
        part = local + ".part"
        for attempt in range(2):
            off = os.path.getsize(part) if os.path.exists(part) else 0
            if off > size:
                off = 0
            with open(part, "r+b" if off else "wb") as f:
                f.seek(off)
                while off < size:
                    n = min(CHUNK, size - off)
                    code = ("import base64,os\np=os.path.expanduser(%r)\nf=open(p,'rb')\nf.seek(%d)\n"
                            "print(base64.b64encode(f.read(%d)).decode())" % (remote, off, n))
                    ok, so, se = self.exec(code, echo=False)
                    if not ok:
                        raise RuntimeError("chunk failed: " + se)
                    data = base64.b64decode(so.strip().splitlines()[-1])
                    if not data:
                        raise RuntimeError("empty chunk")
                    f.write(data)
                    off += len(data)
                    if verbose:
                        print(f"\r{remote}: {off}/{size}", end="", flush=True)
            h = hashlib.sha256()
            with open(part, "rb") as f:
                for b in iter(lambda: f.read(1 << 20), b""):
                    h.update(b)
            if h.hexdigest() == digest:
                os.replace(part, local)
                if verbose:
                    print(f"\nOK {local} sha256={digest}")
                return digest
            os.remove(part)  # corrupt prefix: restart once
        raise SystemExit(f"SHA256 MISMATCH pulling {remote}")

    def pulldir(self, remote_dir, local_dir, pattern="*"):
        """Download every remote file of a directory matching ``pattern``."""
        code = ("import glob,os\nd=os.path.expanduser(%r)\n"
                "for f in sorted(glob.glob(os.path.join(d,%r),recursive=True)):\n"
                "    if os.path.isfile(f): print(f)" % (remote_dir, pattern))
        ok, so, se = self.exec(code, echo=False)
        if not ok:
            raise RuntimeError(se)
        base = remote_dir.rstrip("/")
        for rf in so.split("\n"):
            rf = rf.strip()
            if not rf:
                continue
            rel = os.path.relpath(rf, base).replace("\\", "/")
            self.pull(rf, os.path.join(local_dir, *rel.split("/")))

    def sync_code(self, root, repo="."):
        """Push the local commits the remote checkout does not have (incremental git bundle) and fast-forward it there."""
        def git(*a):
            """Run a git command in the local repository."""
            return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True)
        ok, so, se = self.exec(
            "import subprocess\nr=subprocess.run(['git','-C',%r,'rev-parse','HEAD'],capture_output=True,text=True)\n"
            "print(r.stdout.strip() or ('ERR '+r.stderr))" % root, echo=False)
        remote_head = so.strip().splitlines()[-1] if so.strip() else ""
        local_head = git("rev-parse", "HEAD").stdout.strip()
        print(f"remote HEAD {remote_head[:12]}  local HEAD {local_head[:12]}")
        if remote_head == local_head:
            print("already up to date")
            return
        have = len(remote_head) == 40 and git("cat-file", "-e", remote_head + "^{commit}").returncode == 0
        spec = ["^" + remote_head, "HEAD"] if have else ["HEAD"]
        if not have:
            print("remote HEAD unknown locally; sending full history")
        with tempfile.TemporaryDirectory() as td:
            bundle = os.path.join(td, "code.bundle")
            r = git("bundle", "create", bundle, *spec)
            if r.returncode:
                raise SystemExit("git bundle failed: " + r.stderr)
            self.push(bundle, "/home/marimo/work/code.bundle")
        code = (
            "import subprocess\n"
            "def g(*a):\n"
            "    r=subprocess.run(['git','-C',%r,*a],capture_output=True,text=True); print('$ git',*a); print(r.stdout+r.stderr); return r.returncode\n"
            "assert g('fetch','/home/marimo/work/code.bundle','HEAD')==0\n"
            "assert g('merge','--ff-only','FETCH_HEAD')==0\n"
            "g('log','--oneline','-3')\n" % root)
        ok, _, _ = self.exec(code)
        if not ok:
            raise SystemExit("remote merge failed")


def main():
    """CLI entry: ``exec``, ``push``, ``pushdir``, ``pull``, ``pulldir``, ``sync-code`` (see module docstring)."""
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("MOLAB_URL"))
    ap.add_argument("--token", default=os.environ.get("MOLAB_TOKEN"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("exec")
    e.add_argument("file", nargs="?")
    e.add_argument("-c", dest="code")
    e.add_argument("--timeout", type=int, default=600)
    p = sub.add_parser("push")
    p.add_argument("local")
    p.add_argument("remote")
    d = sub.add_parser("pushdir")
    d.add_argument("local_dir")
    d.add_argument("remote_dir")
    d.add_argument("--glob", default="*")
    q = sub.add_parser("pull")
    q.add_argument("remote")
    q.add_argument("local")
    qd = sub.add_parser("pulldir")
    qd.add_argument("remote_dir")
    qd.add_argument("local_dir")
    qd.add_argument("--glob", default="*")
    sc = sub.add_parser("sync-code")
    sc.add_argument("--root", default="/home/marimo/work/IE643-Novelty")
    sc.add_argument("--repo", default=".")
    a = ap.parse_args()
    if not a.url:
        raise SystemExit("set MOLAB_URL (or --url)")
    m = Molab(a.url, a.token)
    if a.cmd == "exec":
        code = a.code if a.code is not None else (sys.stdin.read() if a.file in (None, "-") else open(a.file, encoding="utf-8").read())
        ok, _, _ = m.exec(code, timeout=a.timeout)
        sys.exit(0 if ok else 1)
    elif a.cmd == "push":
        m.push(a.local, a.remote)
    elif a.cmd == "pull":
        m.pull(a.remote, a.local)
    elif a.cmd == "pulldir":
        m.pulldir(a.remote_dir, a.local_dir, a.glob)
    elif a.cmd == "sync-code":
        m.sync_code(a.root, a.repo)
    else:
        import glob
        for f in sorted(glob.glob(os.path.join(a.local_dir, a.glob))):
            if os.path.isfile(f):
                m.push(f, a.remote_dir.rstrip("/") + "/" + os.path.basename(f))


if __name__ == "__main__":
    main()
