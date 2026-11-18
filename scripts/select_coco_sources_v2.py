#!/usr/bin/env python
"""Select and download the v2 source photographs (COCO 2017 train) -> sources_v2.json.

Runs where the internet is available (Molab). Only the annotations zip
(~252 MB) and the *selected* images (~600 x ~160 KB) are fetched; the 18 GB
train2017 image zip is never downloaded.

    python scripts/select_coco_sources_v2.py --project-root .            # full run
    python scripts/select_coco_sources_v2.py --project-root . --plan-only # list, no image download

Selection (all deterministic given --seed and the COCO files):
  * candidate images per category: has a non-crowd annotation of that category covering
    >= min_category_area_fraction of the image, and min(width,height) >= min_side_px;
  * category round-robin: each wave offers every still-short category its next
    candidate (order = seeded hash); a candidate is accepted only after download and
    checks: decodable, RGB with real colour (mean per-pixel channel std >=
    grayscale_channel_std_max), 8x8 average-hash Hamming distance > ahash_hamming_max to
    every already accepted image and to the v1 images;
  * the 36 v1 sources (match by COCO image id) are always included with their original
    splits and are never re-selected;
  * new sources: the 24 with the smallest sha256(seed|id) become 'development', rest 'train'.

Resumable: downloads are verified against data/v2/sources/_ledger.json (path -> sha256);
files that match are skipped. Re-running reproduces the same selection.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

USER_AGENT = "fpcamera-data-prep/1.0 (research dataset builder)"


# ------------------------------------------------------------------ utilities
def sha256_bytes(b):
    """sha256 of a bytes object."""
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    """Streaming sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_key(seed, *parts):
    """Seeded hash used to order candidates deterministically (selection never depends on set order)."""
    return hashlib.sha256("|".join(str(p) for p in (seed, *parts)).encode()).hexdigest()


def ahash(image: Image.Image) -> int:
    """64-bit 8x8 average hash."""
    small = np.asarray(image.convert("L").resize((8, 8), Image.LANCZOS), dtype=np.float64)
    bits = (small > small.mean()).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    """Hamming distance between two 64-bit average hashes (near-duplicate filter)."""
    return bin(a ^ b).count("1")


def channel_std(image: Image.Image) -> float:
    """Mean per-pixel std across R,G,B (0 for pure grayscale)."""
    if image.mode == "L":
        return 0.0
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    return float(arr.std(axis=2).mean())


def download(url, dest, *, retries=5, timeout=60, fetch=None):
    """Download to dest atomically; returns sha256. `fetch(url)->bytes` is injectable."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(retries):
        try:
            if fetch is not None:
                data = fetch(url)
            else:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = r.read()
            temp = dest.with_name(dest.name + ".partial")
            temp.write_bytes(data)
            os.replace(temp, dest)
            return sha256_bytes(data)
        except Exception as error:  # noqa: BLE001 - retry any network error
            last = error
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Failed to download {url}: {last!r}")


def download_big(url, dest, *, retries=5):
    """Resumable large download (Range) for the annotations zip."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    part = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            os.replace(part, dest)
            return
        except Exception as error:  # noqa: BLE001
            print(f"[annotations] attempt {attempt + 1} failed: {error!r}", flush=True)
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Could not download {url}")


# ------------------------------------------------------------- COCO metadata
def load_coco_train(ann_zip, work_dir):
    """Return (images_by_id, categories, per-image category areas, captions, licenses)."""
    with zipfile.ZipFile(ann_zip) as z:
        with z.open("annotations/instances_train2017.json") as f:
            inst = json.load(f)
        with z.open("annotations/captions_train2017.json") as f:
            caps = json.load(f)
    images = {im["id"]: im for im in inst["images"]}
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    best = {}  # (image_id, cat_id) -> max area fraction among non-crowd annotations
    for a in inst["annotations"]:
        if a.get("iscrowd"):
            continue
        im = images[a["image_id"]]
        frac = a["area"] / float(im["width"] * im["height"])
        k = (a["image_id"], a["category_id"])
        if frac > best.get(k, 0.0):
            best[k] = frac
    captions = {}
    for c in caps["annotations"]:
        captions.setdefault(c["image_id"], []).append(c)
    licenses = {l["id"]: l for l in inst["licenses"]}
    return images, cats, best, captions, licenses


def build_candidate_queues(images, cats, best, *, seed, min_side, min_frac, exclude_ids):
    """Per-category queues of COCO images that pass the size and object-coverage filters, in seeded hash order."""
    queues = {name: [] for name in cats.values()}
    for (image_id, cat_id), frac in best.items():
        im = images[image_id]
        if image_id in exclude_ids or min(im["width"], im["height"]) < min_side or frac < min_frac:
            continue
        queues[cats[cat_id]].append(image_id)
    for name, ids in queues.items():
        ids.sort(key=lambda i: hash_key(seed, "cand", name, i))
    return {k: v for k, v in sorted(queues.items()) if v}


# ------------------------------------------------------------------ selection
def select(
    queues,
    images,
    *,
    target,
    ahash_max,
    gray_std_min,
    seed_hashes,
    fetch_image,
    workers=4,
    log=print,
):
    """Round-robin waves. `fetch_image(image_id)->(path, sha256)` downloads one image.

    Deterministic: downloads in a wave run concurrently but results are evaluated in
    category order, so concurrency cannot change the outcome.
    """
    accepted = []  # dicts
    hashes = list(seed_hashes)
    per_cat = {c: 0 for c in queues}
    cursor = {c: 0 for c in queues}
    rejected = {"grayscale": 0, "duplicate": 0, "decode": 0}
    level = 1
    taken = set()
    while len(accepted) < target:
        active = [c for c in queues if per_cat[c] < level and cursor[c] < len(queues[c])]
        if not active:
            if all(cursor[c] >= len(queues[c]) for c in queues) or level > 1000:
                log("[select] candidate pools exhausted")
                break
            level += 1
            continue
        offered = []
        for c in active:
            while cursor[c] < len(queues[c]) and queues[c][cursor[c]] in taken:
                cursor[c] += 1
            if cursor[c] < len(queues[c]):
                offered.append((c, queues[c][cursor[c]]))
                taken.add(queues[c][cursor[c]])
                cursor[c] += 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda ci: _safe_fetch(fetch_image, ci[1]), offered))
        for (cat, image_id), res in zip(offered, results):
            if len(accepted) >= target:
                break
            if res is None:
                rejected["decode"] += 1
                continue
            path, sha = res
            try:
                with Image.open(path) as im:
                    im.load()
                    gray = channel_std(im) < gray_std_min
                    h = ahash(im)
            except Exception:  # noqa: BLE001
                rejected["decode"] += 1
                continue
            if gray:
                rejected["grayscale"] += 1
                continue
            if any(hamming(h, other) <= ahash_max for other in hashes):
                rejected["duplicate"] += 1
                continue
            hashes.append(h)
            per_cat[cat] += 1
            accepted.append({"image_id": image_id, "category": cat, "path": str(path), "sha256": sha, "ahash": f"{h:016x}"})
        log(f"[select] level {level}: accepted {len(accepted)}/{target} rejected {rejected}")
    return accepted, rejected


def _safe_fetch(fetch_image, image_id):
    try:
        return fetch_image(image_id)
    except Exception as error:  # noqa: BLE001
        print(f"[download] {image_id} failed: {error!r}", flush=True)
        return None


# ---------------------------------------------------------------- v1 handling
def load_v1(project_root, cfg, *, fetch=None, log=print):
    """Return v1 records (re-pathed, with verified files) and their ahashes."""
    sel = cfg["selection"]
    manifest = json.loads((project_root / sel["v1_manifest"]).read_text(encoding="utf-8"))
    out, hashes = [], []
    for rec in manifest["records"]:
        rel = rec["source_image"]
        path = project_root / sel["v1_sources_dir"] / rel
        if not path.exists() or sha256_file(path) != rec["original_image_sha256"]:
            url = rec["provenance"]["coco_url"]
            path = project_root / "data/v2/sources/val2017" / Path(rel).name
            if not path.exists() or sha256_file(path) != rec["original_image_sha256"]:
                log(f"[v1] fetching {url}")
                download(url, path, fetch=fetch)
            if sha256_file(path) != rec["original_image_sha256"]:
                raise RuntimeError(f"v1 source digest mismatch after download: {url}")
        with Image.open(path) as im:
            hashes.append(ahash(im))
        new = dict(rec)
        new["source_image"] = path.relative_to(project_root).as_posix()
        new["origin"] = "v1"
        out.append(new)
    return out, hashes


# ------------------------------------------------------------------------ main
def build_manifest(project_root, cfg, *, seed, target=None, dev=None, fetch=None, plan_only=False,
                   ann_zip=None, workers=None, log=print):
    """Select the sources by deterministic category round-robin (candidates are accepted only after download and checks:
    decodable, real colour, no near-duplicate by 8x8 average hash), always keep the 36 v1 sources, split the new ones into
    train / development, and write ``sources_v2.json`` with licences, hashes and provenance; ``plan_only`` stops before
    downloading.
    """
    sel = cfg["selection"]
    target = target or sel["target_new_sources"]
    dev = sel["new_dev_sources"] if dev is None else dev
    workers = workers or sel["download_workers"]
    work = project_root / "data/v2/_work"
    img_dir = project_root / "data/v2/sources/train2017"
    ledger_path = project_root / "data/v2/sources/_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    lock = threading.Lock()

    v1_records, v1_hashes = ([], [])
    if sel["include_v1_sources"]:
        v1_records, v1_hashes = load_v1(project_root, cfg, fetch=fetch, log=log)
    v1_ids = {r["source_image_id"] for r in v1_records}

    if ann_zip is None:
        ann_zip = work / "annotations_trainval2017.zip"
        log(f"[annotations] ensuring {ann_zip}")
        download_big(sel["annotations_url"], ann_zip)
    images, cats, best, captions, licenses = load_coco_train(ann_zip, work)
    queues = build_candidate_queues(
        images, cats, best, seed=seed, min_side=sel["min_side_px"],
        min_frac=sel["min_category_area_fraction"], exclude_ids=v1_ids,
    )
    log(f"[select] {len(queues)} categories, {sum(len(v) for v in queues.values())} candidate pairs")
    if plan_only:
        return {"categories": {k: len(v) for k, v in queues.items()}}

    def fetch_image(image_id):
        """Download one COCO image (verifying size and hash) into the sources folder."""
        im = images[image_id]
        dest = img_dir / im["file_name"]
        key = dest.relative_to(project_root).as_posix()
        if dest.exists() and ledger.get(key) == sha256_file(dest):
            return dest, ledger[key]
        sha = download(sel["image_url_template"].format(file_name=im["file_name"]), dest, fetch=fetch)
        with lock:
            ledger[key] = sha
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = ledger_path.with_name(ledger_path.name + ".partial")
            tmp.write_text(json.dumps(ledger, sort_keys=True, indent=0))
            os.replace(tmp, ledger_path)
        return dest, sha

    accepted, rejected = select(
        queues, images, target=target, ahash_max=sel["ahash_hamming_max"],
        gray_std_min=sel["grayscale_channel_std_max"], seed_hashes=v1_hashes,
        fetch_image=fetch_image, workers=workers, log=log,
    )
    # deterministic dev split: smallest hash
    ranked = sorted(accepted, key=lambda a: hash_key(seed, "split", a["image_id"]))
    dev_ids = {a["image_id"] for a in ranked[:dev]}
    new_records = []
    for a in sorted(accepted, key=lambda a: a["image_id"]):
        im = images[a["image_id"]]
        lic = licenses[im["license"]]
        new_records.append({
            "source_id": f"coco2017train_{a['image_id']:012d}",
            "source_image_id": a["image_id"],
            "target_category": a["category"],
            "original_image_sha256": a["sha256"],
            "source_image": Path(a["path"]).relative_to(project_root).as_posix(),
            "official_captions": [
                {"image_id": c["image_id"], "id": c["id"], "caption": c["caption"]}
                for c in sorted(captions.get(a["image_id"], []), key=lambda c: c["id"])
            ],
            "training_caption": cfg["sources"]["training_caption"],
            "original_license": {"url": lic["url"], "id": lic["id"], "name": lic["name"]},
            "original_license_id": lic["id"],
            "provenance": {
                "source": "COCO2017 / Flickr",
                "coco_url": im["coco_url"],
                "flickr_url": im.get("flickr_url"),
                "license_url": lic["url"],
                "original_unchanged": True,
                "original_image_metadata": {k: im[k] for k in ("id", "file_name", "width", "height", "date_captured", "license")},
                "purpose": "Project source photograph for synthetic 2D motion supervision; not an observed video",
            },
            "split": "development" if a["image_id"] in dev_ids else "train",
            "origin": "v2",
            "ahash": a["ahash"],
        })
    records = v1_records + new_records
    ids = [r["source_id"] for r in records]
    assert len(set(ids)) == len(ids)
    manifest = {
        "schema_version": 2,
        "extends": "data/flow_poc/pilot_sources.json (schema_version 1): same record keys; source_image is relative to the project root; extra keys: origin, ahash",
        "selection": {
            "dataset": sel["dataset"],
            "seed": seed,
            "method": "category round-robin waves over seeded-hash-ordered candidates; downloaded then filtered",
            "filters": {k: sel[k] for k in ("min_side_px", "min_category_area_fraction", "ahash_hamming_max", "grayscale_channel_std_max")},
            "v1_sources_included": len(v1_records),
            "new_sources": len(new_records),
            "rejected_counts": rejected,
            "split_counts": {s: sum(1 for r in records if r["split"] == s) for s in ("train", "development")},
        },
        "records": records,
    }
    out = project_root / cfg["sources"]["manifest"]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    os.replace(tmp, out)
    log(f"[done] wrote {out}: {len(records)} sources {manifest['selection']['split_counts']}")
    return manifest


def main():
    """CLI entry (see module docstring)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--config", type=Path, default=ROOT / "configs/data_v2.json")
    p.add_argument("--target", type=int, help="override target_new_sources")
    p.add_argument("--dev", type=int, help="override new_dev_sources")
    p.add_argument("--workers", type=int)
    p.add_argument("--annotations-zip", type=Path, help="use an already downloaded annotations zip")
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    build_manifest(
        a.project_root.resolve(), cfg, seed=cfg["seed"], target=a.target, dev=a.dev,
        ann_zip=a.annotations_zip, workers=a.workers, plan_only=a.plan_only,
    )


if __name__ == "__main__":
    main()
