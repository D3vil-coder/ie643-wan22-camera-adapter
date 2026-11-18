#!/usr/bin/env python
"""One-shot recovery of a fresh Molab sandbox (runs LOCALLY, drives the sandbox via molab_sync.Molab).

  export MOLAB_URL=https://sb-<id>.sb.molab.run MOLAB_TOKEN=<token>        # never commit
  python tools/molab_recover.py [--resume-cache-build] [--resume-training RUN] [--only STEP ...] [--plan]
                                  [--resume-eval NAME:DIR ...] [--checkpoints 77,250 [--checkpoints-run RUN]]

Steps (each idempotent: it is skipped when its check passes):
  probe, code, deps, token, model, restore, extras (v1 warm-start adapter + examples/inputs), build (--resume-cache-build), train (--resume-training RUN)
Long remote work is always started detached (nohup, own session) and polled with short calls, because a
blocking exec stream drops on long operations.  The HF token (HF_WRITE_TOKEN) is read locally from the KEY=value file named by
$FPCAMERA_KEYS_ENV (default: <repo root>/keys.env, git-ignored; see tools/local_config.py) and written to $HF_HOME/token (0600) on
the sandbox; it is never printed.  Commit local work first: the repo is shipped as a git bundle of HEAD.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_config import KeysEnvNotFound, keys_env_path  # noqa: E402

WORK = "/home/marimo/work"
ROOT = WORK + "/IE643-Novelty"
HF_HOME = WORK + "/hf"
LOGS = WORK + "/logs"
PIDS = WORK + "/pids"
BRANCH = "D3vil-coder/Main"
CACHE_REL = "data/v2/cache_1280x704_f121_v2"
SESSION_HOURS = 3.0  # Molab hard-kills sandboxes about 3 h after start (default of --session-hours)
DEADLINE_MARGIN_MINUTES = 15  # default of --deadline-margin-minutes
STEPS = ["probe", "code", "deps", "token", "model", "restore", "extras", "build", "train"]
V1_ADAPTER = "artifacts/flow_poc_all/adapter_000300.safetensors"
V1_ADAPTER_SHA256 = "9e2963f264ae5dcd9c406f12e32194782ba523908fb4d2220d8f7774906efdbc"
N_EXAMPLE_INPUTS = 16  # 11 case PNGs + 5 COCO jpgs, kept in the HF dataset repo under examples_inputs/


try:  # remote progress bars contain non-cp1252 glyphs; never crash on logging
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def log(msg):
    """Timestamped print (flushed)."""
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def plan_steps(done: dict, only=None, want_build=False, train_run=None, force=()):
    """Pure planning. ``done``: step -> bool (check result). Returns [(step, 'run'|'skip'|'off')]."""
    out = []
    for s in STEPS:
        if s == "build" and not want_build:
            out.append((s, "off"))
        elif s == "train" and not train_run:
            out.append((s, "off"))
        elif only and s not in only:
            out.append((s, "off"))
        elif done.get(s) and s not in force:
            out.append((s, "skip"))
        else:
            out.append((s, "run"))
    return out


def read_hf_token(path=None):
    """Read ``HF_WRITE_TOKEN`` from the local keys file (the value is never printed or logged).

    ``path`` defaults to ``$FPCAMERA_KEYS_ENV``, else ``<repo root>/keys.env`` (:func:`local_config.keys_env_path`);
    a clear ``SystemExit`` explains how to set it when neither exists.
    """
    if path is None:
        try:
            path = keys_env_path()
        except KeysEnvNotFound as exc:
            raise SystemExit(str(exc)) from None
    for line in open(path, encoding="utf8"):
        if "=" in line:
            k, v = line.split("=", 1)
            if k.strip() == "HF_WRITE_TOKEN":
                return v.strip().strip("\"'")
    raise SystemExit(f"HF_WRITE_TOKEN not found in the keys file {path}")


def parse_name_dir(text):
    """Parse ``NAME:DIR`` (argparse type)."""
    if ":" not in text:
        raise argparse.ArgumentTypeError(f"expected NAME:DIR, got {text!r}")
    name, d = text.split(":", 1)
    if not name or not d:
        raise argparse.ArgumentTypeError(f"expected NAME:DIR, got {text!r}")
    return name, d


def parse_steps(text):
    """Parse ``77,250,477`` into a list of ints (argparse type)."""
    try:
        return [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}") from None


def cmdline_matches(cmdline, needle):
    """True when a /proc cmdline (NUL or space separated) contains ``needle``."""
    return bool(cmdline) and needle in cmdline.replace(chr(0), " ")


def decide_launch(pidfile_pid, pidfile_cmdline, scan_pids, needle):
    """Pure dedup decision. ``pidfile_pid``: int or None (pidfile content); ``pidfile_cmdline``: cmdline of that pid
    or None when it is dead; ``scan_pids``: /proc scan matches for ``needle`` (fallback).
    Returns ('skip', why) when a live process for the target exists, else ('launch', why)."""
    if pidfile_pid is not None and cmdline_matches(pidfile_cmdline, needle):
        return "skip", f"pidfile pid {pidfile_pid} alive"
    if scan_pids:
        return "skip", "process scan found " + ",".join(map(str, scan_pids))
    why = "no pidfile" if pidfile_pid is None else f"stale pidfile (pid {pidfile_pid} dead or different command)"
    return "launch", why


class Remote:
    """Thin wrapper over molab_sync.Molab with retries and shell/detach helpers."""

    def __init__(self, m):
        self.m = m

    def py(self, code, timeout=300, tries=4):
        """Run python code on the sandbox with retries on dropped streams; returns ``(ok, stdout, stderr)``."""
        last = None
        for i in range(tries):
            try:
                ok, so, se = self.m.exec(code, timeout=timeout, echo=False)
                return ok, so.strip(), se
            except Exception as exc:  # noqa: BLE001 - dropped stream / transient HTTP
                last = exc
                log(f"[retry] exec: {type(exc).__name__}: {str(exc)[:100]}")
                time.sleep(5 * (i + 1))
        raise RuntimeError(f"remote exec failed: {last!r}")

    def sh(self, cmd, timeout=300):
        """Run a bash command on the sandbox; returns ``(ok, combined output + RC)``."""
        code = ("import subprocess\nr=subprocess.run(['bash','-c',%r],capture_output=True,text=True)\n"
                "print(r.stdout+r.stderr);print('RC=%%d'%%r.returncode)" % cmd)
        ok, so, se = self.py(code, timeout=timeout)
        rc = int(so.rsplit("RC=", 1)[1]) if "RC=" in so else 1
        return rc, so.rsplit("RC=", 1)[0].strip()

    def venv_sh(self, cmd, timeout=300):
        """Run a bash command inside the sandbox's prepared environment (``env.sh``)."""
        return self.sh(f"source {WORK}/env.sh >/dev/null 2>&1; {cmd}", timeout)

    def detach(self, name, cmd):
        """Start ``cmd`` detached with log LOGS/name.log and marker LOGS/name.rc written on exit.
        The wrapper shell's pid (its cmdline contains ``cmd``) is written to PIDS/name.pid."""
        script = (f"mkdir -p {LOGS} {PIDS}; echo $$ > {PIDS}/{name}.pid; rm -f {LOGS}/{name}.rc; source {WORK}/env.sh >/dev/null 2>&1; "
                  f"({cmd}); echo $? > {LOGS}/{name}.rc")
        code = ("import subprocess\nsubprocess.Popen(['nohup','bash','-c',%r],stdout=open(%r,'ab'),"
                "stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)\nprint('started')"
                % (script, f"{LOGS}/{name}.log"))
        self.py(code)

    def rc(self, name):
        """Exit code of a detached job (None while it is still running)."""
        _, out = self.sh(f"cat {LOGS}/{name}.rc 2>/dev/null")
        return int(out) if out.strip().lstrip("-").isdigit() else None

    def tail(self, name, n=3):
        """Last ``n`` lines of a detached job's log."""
        return self.sh(f"tail -n {n} {LOGS}/{name}.log 2>/dev/null")[1]

    def wait(self, name, every=20, timeout=3 * 3600):
        """Poll a detached job until it exits (or the timeout); returns its exit code."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            rc = self.rc(name)
            if rc is not None:
                return rc
            log(f"[wait {name}] {self.tail(name, 1)[-160:]}")
            time.sleep(every)
        raise SystemExit(f"timeout waiting for {name}")

    def launch_once(self, name, needle, cmd):
        """Detach ``cmd`` unless a live process for the same target exists. Returns True when launched."""
        _, out = self.sh(f"cat {PIDS}/{name}.pid 2>/dev/null")
        pid = int(out) if out.strip().isdigit() else None
        cl = None
        if pid is not None:
            _, cl = self.sh(r"tr '\0' ' ' < /proc/%d/cmdline 2>/dev/null" % pid)
        action, why = decide_launch(pid, cl, self.pids(needle) if not cmdline_matches(cl, needle) else [], needle)
        if action == "skip":
            log(f"[dedup] {name}: not launching ({why})")
            return False
        self.detach(name, cmd)
        log(f"[launch] {name} ({why})")
        return True

    def pids(self, needle):
        """Process ids on the sandbox whose command line contains ``needle`` (ignoring ``bash -c`` wrappers)."""
        _, out = self.sh("for d in /proc/[0-9]*; do c=$(tr '\\0' ' ' < $d/cmdline 2>/dev/null); "
                         f"case \"$c\" in *{needle}*) case \"$c\" in *'bash -c'*) ;; *) echo ${{d#/proc/}};; esac;; esac; done")
        return [p for p in out.split() if p.isdigit()]


class Recovery:
    """Runs the recovery steps in order (probe, code, deps, token, model, restore, extras, build, train).  Every step has a
    ``check`` that says whether it is already done and is skipped then, so the whole recovery is idempotent.  Long work is
    started detached on the sandbox and polled.
    """

    def __init__(self, m, a):
        self.r, self.a = Remote(m), a
        self.m = m
        self.py_bin = None
        self.torch_ver = None
        self.times = {}

    # ---------------------------------------------------------------- checks (True = already done)
    def check(self, step):
        """True if recovery step ``step`` is already satisfied on the sandbox."""
        r = self.r
        if step == "probe":
            return r.sh(f"test -f {WORK}/env.sh && echo ok")[1].endswith("ok")
        if step == "code":
            return r.sh(f"git -C {ROOT} rev-parse HEAD")[1].strip() == self.local_head()
        if step == "deps":
            return r.venv_sh(f"cd {ROOT} && python -c \"from fpcamera.wan import require_pinned_runtime as q; q(); "
                             "import marimo, cv2, lpips\" && echo ok")[1].endswith("ok")
        if step == "token":
            rc, o = r.venv_sh("python -c \"from huggingface_hub import whoami; print(whoami()['name'])\"")
            return rc == 0 or "429" in o or "Too Many Requests" in o
        if step == "model":
            return r.venv_sh("python -c \"from huggingface_hub import snapshot_download as d; "
                             f"d('Wan-AI/Wan2.2-TI2V-5B-Diffusers', revision='{self.revision()}', local_files_only=True); "
                             "print('ok')\"")[1].endswith("ok")
        if step == "restore":
            n, rows = self.cache_counts()
            return rows >= 640 and n >= rows and r.sh(f"test -f {ROOT}/data/v2/sources_v2.json && echo ok")[1].endswith("ok")
        if step == "extras":
            return r.venv_sh(f"cd {ROOT} && test -f {V1_ADAPTER} && test $(ls examples/inputs 2>/dev/null | wc -l) -ge {N_EXAMPLE_INPUTS} "
                             "&& echo ok")[1].endswith("ok")
        return False

    def local_head(self):
        """Commit id of the local checkout (the sandbox gets this commit as a git bundle)."""
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    def revision(self):
        """Pinned Wan base revision to download."""
        return "b8fff7315c768468a5333511427288870b2e9635"

    def cache_counts(self):
        """``(clip files, index rows)`` of the v2 cache on the sandbox."""
        code = (f"import os,glob\nr='{ROOT}/{CACHE_REL}'\nn=len(glob.glob(r+'/clips/*.safetensors'))\n"
                "p=r+'/index.jsonl'\nrows=sum(1 for l in open(p) if l.strip()) if os.path.exists(p) else 0\nprint(n,rows)")
        ok, so, _ = self.r.py(code)
        try:
            n, rows = map(int, so.split()[-2:])
        except ValueError:
            return 0, 0
        return n, rows

    # ---------------------------------------------------------------- steps
    def run_probe(self):
        """Record GPU, CPU count and memory of the sandbox."""
        r = self.r
        _, out = r.sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; nproc; free -g | head -2")
        log("probe:\n" + out)
        ok, so, _ = r.py("import sys,torch\nprint(sys.executable);print(torch.__version__);print(torch.cuda.is_available())")
        py, tv, cu = so.splitlines()[-3:]
        log(f"kernel python {py}, torch {tv}, cuda {cu}")
        self.py_bin, self.torch_ver = py, tv.split("+")[0]
        self.log_age()
        env = (f"export HF_HOME={HF_HOME}\nexport PYTHONPATH={ROOT}/src\nexport PATH={os.path.dirname(py)}:$PATH\n"
               f"cd {ROOT} 2>/dev/null || true\n")
        r.py("import os\nos.makedirs(%r,exist_ok=True);os.makedirs(%r,exist_ok=True)\n"
             "open(%r,'w').write(%r)" % (HF_HOME, LOGS, WORK + "/env.sh", env))

    def log_age(self):
        """Sandbox age from /proc/stat btime (fallback: uptime) and time left under the ~3 h hard kill."""
        code = ("import time\nb=None\nfor l in open('/proc/stat'):\n    if l.startswith('btime'): b=int(l.split()[1])\n"
                "up=float(open('/proc/uptime').read().split()[0])\nnow=time.time()\n"
                "print(b or 0, up, now)")
        try:
            _, so, _ = self.r.py(code)
            btime, up, now = so.split()[-3:]
            btime, up, now = float(btime), float(up), float(now)
            start = btime if btime and 0 < now - btime < 86400 * 2 else now - up
            age = now - start
            left = self.a.session_hours * 3600 - age
            log(f"sandbox start {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start))} UTC, age {age / 60:.0f} min, "
                f"est. remaining {max(left, 0) / 60:.0f} min of a {self.a.session_hours:g} h session")
        except Exception as exc:  # noqa: BLE001
            log(f"[warn] could not measure sandbox age: {exc!r}")

    def ensure_probe_info(self):
        """Make sure the sandbox python/torch information is known (needed by the deps step)."""
        if not self.py_bin:
            ok, so, _ = self.r.py("import sys,torch\nprint(sys.executable);print(torch.__version__)")
            py, tv = so.splitlines()[-2:]
            self.py_bin, self.torch_ver = py, tv.split("+")[0]

    def run_code(self):
        """Ship the local commits: incremental git bundle if the sandbox already has the repo, else a full bundle + clone checked out on
        the local commit (uncommitted changes are not shipped).
        """
        r = self.r
        if r.sh(f"test -d {ROOT}/.git && echo y")[1].endswith("y"):
            self.m.sync_code(ROOT, ".")
            return
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        if dirty:
            log("[warn] local working tree has uncommitted changes; they are NOT shipped")
        with tempfile.TemporaryDirectory() as td:
            b = os.path.join(td, "repo.bundle")
            subprocess.run(["git", "bundle", "create", b, "--all"], check=True, capture_output=True)
            self.m.push(b, WORK + "/repo.bundle")
        rc, out = r.sh(f"cd {WORK} && rm -rf {ROOT} && git clone -q repo.bundle {ROOT} && cd {ROOT} && "
                       f"git checkout -q -B {BRANCH} {self.local_head()} && git log --oneline -1")
        log(out)
        if rc:
            raise SystemExit("clone failed")

    def run_deps(self):
        """Install the requirements into the sandbox environment (torch/torchvision stay at the preinstalled build via a constraints
        file), prefetch the LPIPS backbone and check ``require_pinned_runtime``; detached and polled.
        """
        self.ensure_probe_info()
        code = (
            "import re,pathlib\nR=pathlib.Path(%r)\nout=[]\n"
            "skip=re.compile(r'^(torch|torchvision|numpy|pillow|imageio)(==|>=|\\s|$)',re.I)\n"
            "for f,filt in (('requirements.txt',True),('requirements-ui.txt',False),('requirements-evaluation.txt',True)):\n"
            "    for l in (R/f).read_text().splitlines():\n"
            "        l=l.strip()\n"
            "        if not l or l.startswith('#'): continue\n"
            "        if filt and skip.match(l) and f!='requirements.txt' : continue\n"
            "        if f=='requirements.txt' and re.match(r'^(torch|torchvision)(==|\\s|$)',l,re.I): continue\n"
            "        out.append(l)\n"
            "pathlib.Path(%r).write_text('\\n'.join(dict.fromkeys(out))+'\\n')\nprint(len(out))" % (ROOT, WORK + "/reqs.txt"))
        self.r.py(code)
        # constrain torch/torchvision to the preinstalled build so uv/pip never upgrades them
        rc, out = self.r.sh(
            f"{self.py_bin} - <<'EOF'\nimport torch,torchvision\nprint('torch=='+torch.__version__.split('+')[0])\n"
            f"print('torchvision=='+torchvision.__version__.split('+')[0])\nEOF")
        pins = [x for x in out.splitlines() if "==" in x]
        self.r.sh(f"printf '%s\\n' {' '.join(pins)} > {WORK}/constraints.txt")
        log(f"constraints: {pins}")
        py = self.py_bin
        self.r.detach("deps", f"(command -v uv >/dev/null && uv pip install --python {py} -c {WORK}/constraints.txt -r {WORK}/reqs.txt "
                              f"|| {py} -m pip install -c {WORK}/constraints.txt -r {WORK}/reqs.txt) && echo DEPS_OK")
        if self.r.wait("deps", every=15):
            raise SystemExit("dependency install failed: " + self.r.tail("deps", 15))
        # nice-to-have: pre-fetch the AlexNet backbone LPIPS needs (so evaluation never waits on a download)
        rc, out = self.r.venv_sh("python -c \"import lpips; lpips.LPIPS(net='alex', verbose=False); print('lpips ok')\"", 600)
        log("lpips prefetch: " + (out.splitlines()[-1] if out else f"rc={rc}"))
        rc, out = self.r.venv_sh("python -c \"import torch;print(torch.__version__)\"")
        if out.split("+")[0].strip() != self.torch_ver:
            raise SystemExit(f"torch changed from {self.torch_ver} to {out}: aborting (reinstall torch manually)")
        rc, out = self.r.venv_sh(f"cd {ROOT} && python -c \"from fpcamera.wan import require_pinned_runtime as q; print(q())\"")
        if rc:
            raise SystemExit("require_pinned_runtime failed: " + out[-600:])
        log("require_pinned_runtime OK: " + out[-300:])

    def run_token(self):
        """Write the Hugging Face token to ``$HF_HOME/token`` (mode 0600) on the sandbox; never printed."""
        tok = read_hf_token()
        code = ("import os\nd=%r\nos.makedirs(d,exist_ok=True)\np=d+'/token'\nopen(p,'w').write(%r)\nos.chmod(p,0o600)\nprint('ok')"
                % (HF_HOME, tok))
        self.r.py(code)
        rc, out = self.r.venv_sh("python -c \"from huggingface_hub import whoami; print('hf user', whoami()['name'])\"")
        log(out)
        if rc and "429" not in out and "Too Many Requests" not in out:
            raise SystemExit("HF token rejected")
        if rc:
            log("whoami rate limited (429): token written, continuing")

    def run_model(self):
        """Download the pinned Wan2.2-TI2V-5B-Diffusers revision into the sandbox's HF cache (detached)."""
        self.r.detach("model", "python -u -c \"from huggingface_hub import snapshot_download as d; "
                               f"print(d('Wan-AI/Wan2.2-TI2V-5B-Diffusers', revision='{self.revision()}'))\"")
        if self.r.wait("model", every=15):
            raise SystemExit("model download failed: " + self.r.tail("model", 10))
        log("model ready: " + self.r.tail("model", 1))

    def run_restore(self):
        """Restore sources and the v2 latent cache from the private HF dataset repo (detached)."""
        self.r.detach("restore", f"cd {ROOT} && python -u scripts/restore_from_hf.py --sources --sources-root data/v2 && "
                                 f"python -u scripts/restore_from_hf.py --cache --prefer-shards --cache-dir {CACHE_REL} --sources-root data/v2 --workers 16")
        if self.r.wait("restore", every=20):
            raise SystemExit("restore failed: " + self.r.tail("restore", 15))
        n, rows = self.cache_counts()
        log(f"restore done: {n} clip files, {rows} verified index rows")
        if rows < 640:
            raise SystemExit("expected >= 640 verified rows")

    def run_extras(self):
        """v1 warm-start adapter (sha256-verified) from the HF model repo and examples/inputs from the dataset repo."""
        code = "\n".join([
            "import hashlib, shutil",
            "from pathlib import Path",
            "from huggingface_hub import HfApi, snapshot_download",
            f"root = Path({ROOT!r})",
            "user = HfApi().whoami()['name']",
            f"dst = root / {V1_ADAPTER!r}",
            "if not dst.exists():",
            "    snapshot_download(f'{user}/ie643-camera-adapter-v2', allow_patterns='v1/flow_poc_all/*', "
            "local_dir=str(root / 'artifacts_dl'))",
            "    dst.parent.mkdir(parents=True, exist_ok=True)",
            "    for f in (root / 'artifacts_dl/v1/flow_poc_all').iterdir():",
            "        shutil.copy2(f, dst.parent / f.name)",
            "h = hashlib.sha256(dst.read_bytes()).hexdigest()",
            f"assert h == {V1_ADAPTER_SHA256!r}, ('v1 adapter sha256 mismatch', h)",
            "print('adapter ok', h)",
            "snapshot_download(f'{user}/ie643-camera-v2-cache', repo_type='dataset', allow_patterns='examples_inputs/*', "
            "local_dir=str(root / 'inputs_dl'))",
            "out = root / 'examples/inputs'",
            "out.mkdir(parents=True, exist_ok=True)",
            "for f in (root / 'inputs_dl/examples_inputs').iterdir():",
            "    shutil.copy2(f, out / f.name)",
            "print('inputs', len(list(out.iterdir())))",
        ]) + "\n"
        self.r.py("open(%r,'w').write(%r)" % (WORK + "/extras.py", code))
        rc, out = self.r.venv_sh(f"python {WORK}/extras.py", 900)
        log(out[-400:])
        if rc:
            raise SystemExit("extras (v1 adapter / example inputs) failed")

    def deadline_args(self):
        """Session-deadline flags passed to both build_cache_v2.py and train_v2.py."""
        return (f"--session-hours {self.a.session_hours:g} "
                f"--deadline-margin-minutes {self.a.deadline_margin_minutes:g}")

    def run_build(self):
        """Resume the cache build (``scripts/build_cache_v2.py``) with the session deadline flags."""
        r = self.r
        r.launch_once("build_cache", "scripts/build_cache_v2.py",
                      f"cd {ROOT} && python -u scripts/build_cache_v2.py --project-root . {self.deadline_args()}")
        r.launch_once("backup_cache", "hf_backup.py watch-cache",
                      f"cd {ROOT} && python -u tools/hf_backup.py watch-cache {CACHE_REL} "
                      f"--path-in-repo cache_v2 --sources-root data/v2 --batch-files 25 --batch-seconds 300")

    def run_train(self):
        """Restore the run's checkpoints from HF and (re)launch training (``scripts/train_v2.py``) detached with deadline flags."""
        run, r = self.a.resume_training, self.r
        rc, out = r.venv_sh(f"cd {ROOT} && python scripts/restore_from_hf.py --run {run} --run-dir artifacts/{run}", 1800)
        log(out[-400:])
        if rc and ("nothing under" in out or "EntryNotFoundError" in out):
            log(f"no run '{run}' on HF yet: starting fresh from --init-from")
        elif rc:
            raise SystemExit("run restore failed")
        r.venv_sh(f"cd {ROOT} && test -f artifacts/flow_poc_all/adapter_000300.safetensors || python -c \""
                  "from huggingface_hub import snapshot_download as d;d('D3vil-Reap3r/ie643-camera-adapter-v2',"
                  "allow_patterns='v1/flow_poc_all/*',local_dir='artifacts_dl')\" && mkdir -p artifacts/flow_poc_all && "
                  "cp -n artifacts_dl/v1/flow_poc_all/* artifacts/flow_poc_all/", 900)
        r.launch_once(f"train_{run}", "scripts/train_v2.py",
                      f"cd {ROOT} && python -u scripts/train_v2.py --config configs/train_v2.json --run-dir artifacts/{run} "
                      f"--cache {CACHE_REL}/index.jsonl:1.0 "
                      f"--init-from artifacts/flow_poc_all/adapter_000300.safetensors "
                      f"--allow-identity-change caches {self.deadline_args()}")
        r.launch_once(f"backup_{run}", "hf_backup.py watch-run",
                      f"cd {ROOT} && python -u tools/hf_backup.py watch-run artifacts/{run} --path-in-repo runs/{run}")

    def run_resume_eval(self):
        """Restore eval/<name> dirs from HF and start a watch-dir daemon for each (idempotent)."""
        r = self.r
        for name, d in self.a.resume_eval:
            rc, out = r.venv_sh(f"cd {ROOT} && python -u scripts/restore_from_hf.py --eval {name}:{d} --workers 8", 1800)
            log(out[-300:])
            if rc:
                raise SystemExit(f"eval restore failed for {name}")
            r.launch_once(f"backup_eval_{name}", f"hf_backup.py watch-dir {d} ",
                          f"cd {ROOT} && python -u tools/hf_backup.py watch-dir {d} "
                          f"--repo-type dataset --path-in-repo eval/{name}")

    def run_checkpoints(self):
        """Restore specific historical checkpoints (adapter only; latest.json untouched) for validation/eval."""
        run = self.a.checkpoints_run or self.a.resume_training
        steps = ",".join(map(str, self.a.checkpoints))
        rc, out = self.r.venv_sh(f"cd {ROOT} && python -u scripts/restore_from_hf.py --run {run} "
                                 f"--run-dir artifacts/{run} --checkpoints {steps}", 1800)
        log(out[-500:])
        if rc:
            raise SystemExit(f"checkpoint restore failed for steps {steps}")

    # ---------------------------------------------------------------- driver
    def go(self):
        """Plan and execute all steps; prints a timing summary."""
        a = self.a
        done = {}
        if True:
            for s in ("probe", "code", "deps", "token", "model", "restore", "extras"):
                try:
                    done[s] = self.check(s)
                except Exception as exc:  # noqa: BLE001 - a failing check means "not done"
                    log(f"[check {s}] failed ({type(exc).__name__}); will run")
                    done[s] = False
        plan = plan_steps(done, a.only, a.resume_cache_build, a.resume_training, a.force or ())
        log("plan: " + ", ".join(f"{s}={m}" for s, m in plan))
        if a.plan:
            return 0
        for s, mode in plan:
            if mode != "run":
                continue
            t0 = time.time()
            log(f"=== step {s} ===")
            getattr(self, "run_" + s)()
            self.times[s] = time.time() - t0
            log(f"=== step {s} done in {self.times[s]:.0f}s ===")
        if a.checkpoints:
            self.run_checkpoints()
        if a.resume_eval:
            self.run_resume_eval()
        self.summary()
        return 0

    def summary(self):
        """Print the timing and status summary of the executed steps."""
        r = self.r
        log("---- status summary ----")
        for s, t in self.times.items():
            log(f"  {s:8s} {t:6.0f}s")
        n, rows = self.cache_counts()
        log(f"  cache: {n} clip files, {rows} index rows")
        for label, needle, logname in (("build_cache_v2", "scripts/build_cache_v2.py", "build_cache"),
                                       ("cache watcher", "hf_backup.py watch-cache", "backup_cache"),
                                       ("train_v2", "scripts/train_v2.py", None),
                                       ("run watcher", "hf_backup.py watch-run", None)):
            p = r.pids(needle)
            if p:
                log(f"  {label}: PIDs {','.join(p)}" + (f"  log {LOGS}/{logname}.log" if logname else f"  logs in {LOGS}/"))
                if logname:
                    log("    " + r.tail(logname, 2).replace("\n", "\n    "))


def main(argv=None):
    """CLI entry (see module docstring); needs ``MOLAB_URL`` / ``MOLAB_TOKEN`` in the environment."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("MOLAB_URL"))
    ap.add_argument("--token", default=os.environ.get("MOLAB_TOKEN"))
    ap.add_argument("--resume-cache-build", action="store_true")
    ap.add_argument("--resume-training", metavar="RUN")
    ap.add_argument("--resume-eval", type=parse_name_dir, action="append", default=[], metavar="NAME:DIR",
                    help="restore eval/NAME from HF into DIR (relative to the repo) and start its watch-dir daemon (repeatable)")
    ap.add_argument("--checkpoints", type=parse_steps, default=[], metavar="STEP,STEP",
                    help="restore these historical checkpoints (adapter only, latest.json untouched) of the run")
    ap.add_argument("--checkpoints-run", metavar="RUN", help="run name for --checkpoints (default: --resume-training)")
    ap.add_argument("--session-hours", type=float, default=SESSION_HOURS,
                    help="Molab sandbox lifetime passed to build/train as --session-hours (default 3.0)")
    ap.add_argument("--deadline-margin-minutes", type=float, default=DEADLINE_MARGIN_MINUTES,
                    help="safety margin passed as --deadline-margin-minutes (default 15)")
    ap.add_argument("--only", nargs="+", choices=STEPS)
    ap.add_argument("--force", nargs="+", choices=STEPS, help="re-run these steps even if their check passes")
    ap.add_argument("--plan", action="store_true", help="run the checks and print the plan only")
    a = ap.parse_args(argv)
    if a.checkpoints and not (a.checkpoints_run or a.resume_training):
        ap.error("--checkpoints needs --checkpoints-run (or --resume-training)")
    if not a.url:
        raise SystemExit("set MOLAB_URL (or --url)")
    from molab_sync import Molab
    return Recovery(Molab(a.url, a.token), a).go()


if __name__ == "__main__":
    sys.exit(main())
