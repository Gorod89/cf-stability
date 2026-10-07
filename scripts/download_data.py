"""Download the open datasets into data/raw (run only with the permission of the data owner's licence).

    python scripts/download_data.py openacc follownet waymo
    python scripts/download_data.py ngsim --locations i-80 us-101

Every file is recorded in data/raw/<dataset>/SOURCES.json (url, size, sha256, licence, date).
Existing files of the right size are not downloaded again.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cf_stability.utils import read_json, write_json  # noqa: E402

OPENACC_BASE = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/TransportExpData/JRCDBT0001/LATEST/"
OPENACC_CAMPAIGNS = ("AstaZero", "Casale", "Cherasco", "JRC low speed", "Vicolungo", "ZalaZone")
FOLLOWNET_ARTICLES = {  # figshare article ids of the event sets (collection 6777810, CC0)
    "highd": 23898534,
    "ngsim_i80": 23898519,
    "waymo": 23898513,
}
MIN_FREE_BYTES = 2 * 1024**3


def _check_space(target: Path, needed: int) -> None:
    free = shutil.disk_usage(target).free
    if free - needed < MIN_FREE_BYTES:
        raise SystemExit(f"not enough disk space: {free / 1e9:.1f} GB free, {needed / 1e9:.2f} GB needed, 2 GB reserve")


def _download(session: requests.Session, url: str, dest: Path, expected_size: int | None = None) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        sha = hashlib.sha256(dest.read_bytes()).hexdigest()
        return {"status": "present", "size": dest.stat().st_size, "sha256": sha}
    _check_space(dest.parent, expected_size or 0)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(4):
        try:
            sha = hashlib.sha256()
            size = 0
            with session.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        sha.update(chunk)
                        size += len(chunk)
            if expected_size is not None and size != expected_size:
                raise IOError(f"size mismatch for {url}: {size} != {expected_size}")
            tmp.replace(dest)
            return {"status": "downloaded", "size": size, "sha256": sha.hexdigest()}
        except (requests.RequestException, IOError) as exc:
            if attempt == 3:
                raise
            print(f"  retry {attempt + 1} for {dest.name}: {exc}")
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def _record(root: Path, entries: dict, licence: str, source: str) -> None:
    path = root / "SOURCES.json"
    payload = read_json(path) if path.exists() else {"files": {}}
    payload.update({"licence": licence, "source": source, "updated": datetime.now(timezone.utc).isoformat()})
    payload["files"].update(entries)
    write_json(path, payload)


def _list_index(session: requests.Session, url: str) -> list[str]:
    html = session.get(url, timeout=120).text
    hrefs = re.findall(r'href="([^"?][^"]*)"', html)
    return [h for h in hrefs if not h.startswith(("/", "..", "http", "#"))]


def download_openacc(root: Path, campaigns: tuple[str, ...]) -> None:
    session = requests.Session()
    entries = {}
    for campaign in campaigns:
        base = OPENACC_BASE + quote(campaign) + "/"
        names = [h for h in _list_index(session, base) if not h.endswith("/")]
        print(f"OpenACC {campaign}: {len(names)} files")
        for href in names:
            name = unquote(href)
            info = _download(session, base + href, root / campaign / name)
            entries[f"{campaign}/{name}"] = {"url": base + href, **info}
    _record(root, entries, "CC BY 4.0", OPENACC_BASE)
    total = sum(e["size"] for e in entries.values())
    print(f"OpenACC: {len(entries)} files, {total / 1e6:.1f} MB in {root}")


def download_follownet(root: Path, subsets: tuple[str, ...]) -> None:
    session = requests.Session()
    entries = {}
    for subset in subsets:
        article = session.get(f"https://api.figshare.com/v2/articles/{FOLLOWNET_ARTICLES[subset]}", timeout=120).json()
        for f in article["files"]:
            info = _download(session, f["download_url"], root / f["name"], expected_size=int(f["size"]))
            entries[f["name"]] = {"url": f["download_url"], "doi": article["doi"], "subset": subset, **info}
            print(f"FollowNet {subset}: {f['name']} {info['size'] / 1e6:.1f} MB ({info['status']})")
    _record(root, entries, "CC0", "https://doi.org/10.6084/m9.figshare.c.6777810")


NGSIM_RESOURCE = "https://data.transportation.gov/resource/8ect-6jqj.csv"
NGSIM_INT = ("vehicle_id", "frame_id", "total_frames", "global_time", "v_class", "lane_id", "preceding", "following")
NGSIM_FLOAT = ("local_x", "local_y", "global_x", "global_y", "v_length", "v_width", "v_vel", "v_acc",
               "space_headway", "time_headway")
NGSIM_TEXT = ("o_zone", "d_zone", "int_id", "section_id", "direction", "movement", "location")


def fetch_ngsim(location: str, out_dir: Path, page_size: int = 500_000) -> Path:
    """Page through the Socrata API and write data/raw/ngsim/ngsim_<location>_raw.parquet (no CSV on disk)."""
    import io

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = out_dir / f"ngsim_{location}_raw.parquet"
    if out.exists():
        print(f"NGSIM {location}: {out.name} present")
        return out
    schema = pa.schema(
        [(c, pa.int64()) for c in NGSIM_INT] + [(c, pa.float64()) for c in NGSIM_FLOAT] + [(c, pa.string()) for c in NGSIM_TEXT]
    )
    session = requests.Session()
    tmp = out.with_name(out.name + ".part")
    offset, rows = 0, 0
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        while True:
            params = {"$where": f"location='{location}'", "$order": ":id", "$limit": page_size, "$offset": offset}
            for attempt in range(5):
                try:
                    resp = session.get(NGSIM_RESOURCE, params=params, timeout=600)
                    resp.raise_for_status()
                    break
                except requests.RequestException as exc:
                    if attempt == 4:
                        raise
                    print(f"  retry {attempt + 1} at offset {offset}: {exc}")
                    time.sleep(10 * (attempt + 1))
            page = pd.read_csv(io.BytesIO(resp.content), dtype=str, keep_default_na=False)
            if page.empty:
                break
            page.columns = [c.lower() for c in page.columns]
            data = {}
            for c in NGSIM_INT + NGSIM_FLOAT:
                num = pd.to_numeric(page[c].str.replace(",", "", regex=False), errors="coerce")
                data[c] = num.fillna(0).astype("int64") if c in NGSIM_INT else num.astype("float64")
            for c in NGSIM_TEXT:
                data[c] = page[c].astype(object)
            writer.write_table(pa.Table.from_pandas(pd.DataFrame(data), schema=schema, preserve_index=False))
            rows += len(page)
            offset += page_size
            print(f"NGSIM {location}: {rows} rows", flush=True)
            if len(page) < page_size:
                break
    tmp.replace(out)
    return out


def download_waymo(root: Path) -> None:
    """Car-following pairs of Hu et al. (2022), Mendeley Data 10.17632/wfn2c3437n.2, folder 'Version 2'."""
    session = requests.Session()
    api = "https://data.mendeley.com/public-api/datasets/wfn2c3437n"
    resp = session.get(f"{api}/folders/2", timeout=120)
    if resp.status_code == 403:
        # data.mendeley.com puts a browser challenge in front of some HTTP clients (seen 2026-09-27 for
        # python-requests); it is not worked around here
        raise SystemExit(
            "Mendeley Data refused the request (HTTP 403, browser challenge). Download the five files of the "
            "folder 'Version 2' of https://doi.org/10.17632/wfn2c3437n.2 by hand into " + str(root)
        )
    folders = resp.json()
    folder_id = next(f["id"] for f in folders if f["name"].strip() == "Version 2")
    files = session.get(f"{api}/files", params={"folder_id": folder_id, "version": 2}, timeout=120).json()
    entries = {}
    for f in files:
        details = f["content_details"]
        info = _download(session, details["download_url"], root / f["filename"], expected_size=int(f["size"]))
        if info["sha256"] != details["sha256_hash"]:
            raise IOError(f"sha256 mismatch for {f['filename']}")
        entries[f["filename"]] = {"url": details["download_url"], **info}
        print(f"Waymo pairs: {f['filename']} {info['size'] / 1e6:.1f} MB ({info['status']})")
    _record(root, entries, "CC BY 4.0", "https://doi.org/10.17632/wfn2c3437n.2")


def download_ngsim(root: Path, locations: tuple[str, ...]) -> None:
    for location in locations:
        _check_space(root, 400 * 1024**2)
        path = fetch_ngsim(location, root)
        entry = {path.name: {"url": "https://data.transportation.gov/d/8ect-6jqj", "location": location,
                             "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
        _record(root, entry, "public domain (US DOT)", "https://data.transportation.gov/d/8ect-6jqj")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+", choices=["openacc", "follownet", "ngsim", "waymo"])
    parser.add_argument("--root", default=str(REPO_ROOT / "data" / "raw"))
    parser.add_argument("--campaigns", nargs="+", default=list(OPENACC_CAMPAIGNS))
    parser.add_argument("--subsets", nargs="+", default=list(FOLLOWNET_ARTICLES), choices=list(FOLLOWNET_ARTICLES))
    parser.add_argument("--locations", nargs="+", default=["i-80", "us-101"])
    args = parser.parse_args()
    root = Path(args.root)
    for dataset in args.datasets:
        target = root / dataset
        target.mkdir(parents=True, exist_ok=True)
        if dataset == "openacc":
            download_openacc(target, tuple(args.campaigns))
        elif dataset == "follownet":
            download_follownet(target, tuple(args.subsets))
        elif dataset == "waymo":
            download_waymo(target)
        else:
            download_ngsim(target, tuple(args.locations))


if __name__ == "__main__":
    main()
