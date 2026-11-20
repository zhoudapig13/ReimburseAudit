"""Download the pinned public checkpoint with resumable, verified HTTP ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

import requests
import yaml


def request_with_retry(method, url, **kwargs):
    for attempt in range(12):
        try:
            response = requests.request(method, url, timeout=30, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            print(
                json.dumps(
                    {
                        "stage": "metadata_connection_retry",
                        "host": urlsplit(url).hostname,
                        "attempt": attempt + 1,
                        "error_type": type(error).__name__,
                    }
                ),
                flush=True,
            )
            if attempt == 11:
                raise RuntimeError(
                    f"metadata request exhausted retries at {urlsplit(url).hostname}: {type(error).__name__}"
                ) from None
            time.sleep(min(30, 2 ** min(attempt, 4)) + random.random())
    raise RuntimeError("unreachable")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base512.yaml"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/Qwen3-VL-8B-Instruct")
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument(
        "--record-path", type=Path, default=Path("runs/bootstrap/model_download.json")
    )
    parser.add_argument("--root-files-only", action="store_true")
    parser.add_argument("--transfer-retries", type=int, default=12)
    parser.add_argument("--include-file", action="append", default=[])
    parser.add_argument("--weight-base-url")
    args = parser.parse_args()
    if args.transfer_retries < 1:
        parser.error("transfer-retries must be positive")
    config = yaml.safe_load(args.config.read_text())
    name, revision = config["model"]["name"], config["model"]["revision"]
    cached_manifest = args.output_dir / ".range_parts/manifest.json"
    if cached_manifest.is_file():
        manifest = json.loads(cached_manifest.read_text())
    else:
        response = request_with_retry(
            "GET",
            f"https://huggingface.co/api/models/{name}/revision/{revision}?blobs=true",
        )
        manifest = response.json()
    if manifest["sha"] != revision:
        raise ValueError("resolved checkpoint revision mismatch")
    files = [
        s
        for s in manifest["siblings"]
        if not s["rfilename"].endswith((".md", ".png", ".jpg"))
        and s["rfilename"] != ".gitattributes"
        and (not args.root_files_only or "/" not in s["rfilename"])
        and (not args.include_file or s["rfilename"] in args.include_file)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    work = args.output_dir / ".range_parts"
    work.mkdir(exist_ok=True)
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2))
    urls = {}
    resolve_urls = {}
    refresh_lock = threading.Lock()
    last_refresh = {}
    pending = []
    tasks_by_file = []
    for record in files:
        filename = record["rfilename"]
        path = args.output_dir / filename
        if path.is_file() and path.stat().st_size == record["size"]:
            if not record.get("lfs") or sha256_file(path) == record["lfs"]["sha256"]:
                continue
        url = f"https://huggingface.co/{name}/resolve/{revision}/{filename}"
        if not record.get("lfs"):
            r = request_with_retry("GET", url)
            if len(r.content) != record["size"]:
                raise ValueError(f"small file size mismatch: {filename}")
            path.write_bytes(r.content)
            continue
        weight_url = (
            args.weight_base_url.rstrip("/") + "/" + filename
            if args.weight_base_url
            else url
        )
        r = request_with_retry("HEAD", weight_url, allow_redirects=True)
        urls[filename] = r.url
        resolve_urls[filename] = weight_url
        parts = work / filename
        parts.mkdir(exist_ok=True)
        size = args.chunk_mib * 1024**2
        file_pending = []
        for index, start in enumerate(range(0, record["size"], size)):
            end = min(start + size, record["size"]) - 1
            part = parts / f"{index:06d}"
            if not part.exists() or part.stat().st_size != end - start + 1:
                file_pending.append((filename, start, end, record["size"], part))
        tasks_by_file.append(file_pending)
    # Spread active transfers across independent files rather than saturating
    # one object with the entire connection pool.
    for index in range(max((len(tasks) for tasks in tasks_by_file), default=0)):
        pending.extend(tasks[index] for tasks in tasks_by_file if index < len(tasks))
    started = time.monotonic()
    completed_bytes = 0
    worker_state = threading.local()

    def transfer(item):
        filename, start, end, size, path = item
        temporary = path.with_suffix(".tmp")
        if not hasattr(worker_state, "session"):
            worker_state.session = requests.Session()
        for attempt in range(args.transfer_retries):
            try:
                already = temporary.stat().st_size if temporary.exists() else 0
                if already == end - start + 1:
                    temporary.replace(path)
                    return end - start + 1
                if already > end - start + 1:
                    raise ValueError("invalid partial range length")
                resume_start = start + already
                with worker_state.session.get(
                    urls[filename],
                    headers={"Range": f"bytes={resume_start}-{end}"},
                    stream=True,
                    timeout=(20, 90),
                ) as response:
                    if response.status_code == 403:
                        with refresh_lock:
                            if time.monotonic() - last_refresh.get(filename, 0) > 60:
                                refreshed = request_with_retry(
                                    "HEAD",
                                    resolve_urls[filename],
                                    allow_redirects=True,
                                    params={
                                        "download": "true",
                                        "cache_bust": int(time.time()),
                                    },
                                )
                                urls[filename] = refreshed.url
                                last_refresh[filename] = time.monotonic()
                    response.raise_for_status()
                    if (
                        response.status_code != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {resume_start}-{end}/{size}"
                    ):
                        raise ValueError(
                            f"invalid range response for {filename} part {start}"
                        )
                    with temporary.open("ab") as stream:
                        for chunk in response.iter_content(1024**2):
                            stream.write(chunk)
                if temporary.stat().st_size != end - start + 1:
                    raise ValueError("incomplete range transfer")
                temporary.replace(path)
                return end - start + 1
            except (requests.RequestException, ValueError) as error:
                hint = next(
                    (
                        message
                        for message in [
                            "UNEXPECTED_EOF_WHILE_READING",
                            "CERTIFICATE_VERIFY_FAILED",
                            "Connection reset",
                            "Read timed out",
                        ]
                        if message in str(error)
                    ),
                    None,
                )
                print(
                    json.dumps(
                        {
                            "retry": attempt + 1,
                            "file": filename,
                            "part_start": start,
                            "error_type": type(error).__name__,
                            "error_hint": hint,
                        }
                    ),
                    flush=True,
                )
                if attempt == args.transfer_retries - 1:
                    raise RuntimeError(
                        f"range transfer exhausted retries: {filename} part {start}; {type(error).__name__}"
                    ) from None
                worker_state.session.close()
                worker_state.session = requests.Session()
                if (
                    isinstance(error, requests.ConnectionError)
                    and (attempt + 1) % 3 == 0
                ):
                    with refresh_lock:
                        if time.monotonic() - last_refresh.get(filename, 0) > 30:
                            try:
                                fresh = requests.head(
                                    resolve_urls[filename],
                                    allow_redirects=True,
                                    timeout=15,
                                    params={
                                        "download": "true",
                                        "cache_bust": int(time.time()),
                                    },
                                )
                                fresh.raise_for_status()
                                urls[filename] = fresh.url
                                last_refresh[filename] = time.monotonic()
                            except requests.RequestException:
                                pass
                time.sleep(min(30, 2 ** min(attempt, 4)) + random.random())
        raise RuntimeError("unreachable")

    print(
        json.dumps(
            {"revision": revision, "range_count": len(pending), "workers": args.workers}
        ),
        flush=True,
    )
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(transfer, item) for item in pending]
        failures = []
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                completed_bytes += future.result()
            except RuntimeError as error:
                failures.append(str(error))
            if index % 8 == 0 or index == len(futures):
                print(
                    json.dumps(
                        {
                            "parts_completed": index,
                            "parts_total": len(futures),
                            "downloaded_gib": round(completed_bytes / 1024**3, 3),
                            "elapsed_seconds": round(time.monotonic() - started),
                        }
                    ),
                    flush=True,
                )
        if failures:
            (work / "failed_ranges.json").write_text(
                json.dumps(failures, indent=2) + "\n"
            )
            raise RuntimeError(
                f"{len(failures)} ranges incomplete; completed parts retained for resumption"
            )
    for record in files:
        if not record.get("lfs"):
            continue
        path = args.output_dir / record["rfilename"]
        if not path.is_file() or sha256_file(path) != record["lfs"]["sha256"]:
            temporary = path.with_suffix(".assembling")
            with temporary.open("wb") as stream:
                for part in sorted((work / record["rfilename"]).iterdir()):
                    if part.suffix:
                        continue
                    with part.open("rb") as source:
                        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
                            stream.write(chunk)
            if (
                temporary.stat().st_size != record["size"]
                or sha256_file(temporary) != record["lfs"]["sha256"]
            ):
                raise ValueError(f"checkpoint hash mismatch: {record['rfilename']}")
            temporary.replace(path)
        print("verified", record["rfilename"], flush=True)
    verified = {
        "model": name,
        "revision": revision,
        "local_path": str(args.output_dir.resolve()),
        "transport": "http_range",
        "all_shard_sha256_verified": True,
    }
    args.record_path.parent.mkdir(parents=True, exist_ok=True)
    args.record_path.write_text(json.dumps(verified, indent=2))
    print("download_complete", flush=True)


if __name__ == "__main__":
    main()
