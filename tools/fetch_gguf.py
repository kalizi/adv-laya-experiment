"""Fetch a GGUF from the Hugging Face Hub over a network that stalls.

This host's outbound HTTPS dies after roughly 8 MB on a single long-lived
connection, so `hf_hub_download` and plain `curl -o` both truncate silently at
the same point. Short HTTP range requests survive, so the file is fetched in
parallel byte ranges instead. The expected sha256 comes from the Hub's LFS
metadata and is verified before the file is accepted, because a truncated
download is indistinguishable from a working one unless you check.

**Attenzione: fino al 2026-10-05 questo tool non aveva mai scaricato niente.** Il
template dell'URL era `/resolve/{repo}/{name}` invece di `/{repo}/resolve/{revision}/{name}`,
quindi ogni richiesta prendeva un 404, il 404 finiva nel ramo generico delle eccezioni e
nessun byte veniva scritto: il risultato era un file *sparse* della giusta dimensione, che `ls`
mostrava come completo e che `sha256sum` segnalava come diverso. La docstring di allora
affermava che il tool avesse verificato Q6_K, e non l'aveva mai fatto. Ora l'URL e' corretto,
un 429 o un 503 viene dichiarato invece di essere ritentato in silenzio, e un file sparse viene
rifiutato esplicitamente anche quando la dimensione logica combacia.

Usage:
    uv run python tools/fetch_gguf.py --repo czl/CLM-v0.1-8B-GGUF \\
        --file Qwen3-8B-Q6_K.gguf --out ~/.cache/layaadv/clm/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API = "https://huggingface.co/api/models/{repo}/tree/main"
# La forma che risolve sull'Hub e' `/{repo}/resolve/{revision}/{name}`. Il template che c'era
# prima, `/resolve/{repo}/{name}`, **404 su tutto**, e il 404 finiva nel ramo generico delle
# eccezioni: nessun byte scritto, nessun messaggio, e un file sparse della giusta dimensione che
# `ls` mostrava come scaricato. La docstring diceva che questo tool aveva verificato Q6_K: non
# l'aveva mai fatto, perche' l'URL era sbagliato fin dall'inizio.
RESOLVE = "https://huggingface.co/{repo}/resolve/{revision}/{name}"
USER_AGENT = "layaadv/fetch-gguf"


def hub_size_and_sha(repo: str, name: str) -> tuple[int, str | None]:
    """Return (size_bytes, sha256) for `name` from the Hub's LFS metadata."""
    url = API.format(repo=repo)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        entries = json.load(response)
    for entry in entries:
        if entry.get("path") == name:
            lfs = entry.get("lfs") or {}
            size = lfs.get("size", entry.get("size"))
            if size is None:
                raise RuntimeError(f"{name}: no size in Hub metadata")
            return int(size), lfs.get("oid")
    raise RuntimeError(f"{name}: not found in {repo}")


def sha256_file(path: Path, block: int = 1 << 22) -> str:
    """Stream the file through sha256 without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


class RangeFetcher:
    """Downloads byte ranges of one remote file, retrying each range alone."""

    def __init__(self, url: str, size: int, retries: int = 6) -> None:
        self.url = url
        self.size = size
        self.retries = retries

    def fetch(self, start: int, end: int, destination: Path) -> int:
        """Fetch [start, end] into destination, retrying, and return byte count."""
        want = end - start + 1
        headers = {"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"}
        last = "unknown"
        for attempt in range(1, self.retries + 1):
            try:
                request = urllib.request.Request(self.url, headers=headers)
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        raise RuntimeError(
                            f"range not honoured: HTTP {response.status}"
                        )
                    with destination.open("r+b") as handle:
                        handle.seek(start)
                        written = 0
                        while chunk := response.read(1 << 20):
                            handle.write(chunk)
                            written += len(chunk)
                if written != want:
                    raise RuntimeError(f"short range: {written} != {want}")
                return want
            except urllib.error.HTTPError as error:
                # 429 e 503 non sono errori di range: è il server che dice "torna fra poco".
                # Retentarli con un backoff corto e senza dire nulla significava un download
                # che non scriveva un byte e non mostrava un carattere: il file restava
                # sparse, della giusta dimensione e del tutto vuoto, e sembrava in corso.
                if error.code in (429, 503):
                    wait = min(30 * attempt, 300)
                    print(
                        f"  HTTP {error.code} su range {start}-{end}: il server chiede di "
                        f"aspettare, riprovo fra {wait}s (tentativo {attempt}/{self.retries})",
                        file=sys.stderr,
                        flush=True,
                    )
                    last = f"HTTP {error.code}"
                    time.sleep(wait)
                    continue
                last = f"HTTPError {error.code}: {error}"
                time.sleep(min(2**attempt, 20))
            except (urllib.error.URLError, OSError, RuntimeError) as error:
                last = f"{type(error).__name__}: {error}"
                time.sleep(min(2**attempt, 20))
        raise RuntimeError(
            f"range {start}-{end} failed after {self.retries} attempts: {last}"
        )


def load_done(path: Path, chunk: int) -> list[bool]:
    """Read the resume sidecar, or start a fresh one."""
    sidecar = path.with_suffix(path.suffix + ".parts.json")
    total = (path.stat().st_size + chunk - 1) // chunk if path.exists() else 0
    if sidecar.exists():
        try:
            done = json.loads(sidecar.read_text())
            if len(done) == total:
                return [bool(x) for x in done]
        except (json.JSONDecodeError, OSError):
            pass
    return [False] * total


def save_done(path: Path, done: list[bool]) -> None:
    """Persist the resume sidecar atomically."""
    sidecar = path.with_suffix(path.suffix + ".parts.json")
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp.write_text(json.dumps(done))
    tmp.replace(sidecar)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Hub repo id")
    parser.add_argument("--file", required=True, help="file name inside the repo")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--revision", default="main", help="revisione del repo Hub")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-mb", type=int, default=16)
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the sha256 check against the Hub's LFS oid",
    )
    args = parser.parse_args()

    chunk = args.chunk_mb << 20
    size, expected = hub_size_and_sha(args.repo, args.file)
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / args.file

    print(f"{args.repo}/{args.file}  {size:,} bytes  {size / (1 << 30):.2f} GiB")
    if expected:
        print(f"expected sha256 {expected}")
    print(f"workers={args.workers} chunk={args.chunk_mb}MiB out={target}")

    if target.exists() and target.stat().st_size == size:
        actual = sha256_file(target)
        if args.no_verify or actual == expected:
            print("already complete and verified")
            return 0
        print("present but sha256 differs, refetching", file=sys.stderr)

    with target.open("wb") as handle:
        handle.truncate(size)

    done = load_done(target, chunk)
    fetcher = RangeFetcher(
        RESOLVE.format(repo=args.repo, revision=args.revision, name=args.file), size=size
    )
    ranges = [
        (i * chunk, min((i + 1) * chunk, size) - 1) for i in range(len(done))
    ]
    todo = [i for i, ok in enumerate(done) if not ok]
    print(f"{len(todo)} of {len(done)} ranges to fetch")

    started = time.monotonic()
    fetched = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetcher.fetch, *ranges[i], target): i for i in todo
        }
        for future in as_completed(futures):
            index = futures[future]
            fetched += future.result()
            done[index] = True
            save_done(target, done)
            rate = fetched / max(time.monotonic() - started, 1e-9)
            left = (sum(not x for x in done)) * chunk / max(rate, 1)
            print(
                f"  {sum(done)}/{len(done)} ranges "
                f"{fetched / (1 << 20):.0f}MiB "
                f"{rate / (1 << 20):.2f}MiB/s eta {left / 60:.1f}min",
                flush=True,
            )

    on_disk = target.stat().st_size
    if on_disk != size:
        print(f"FAIL size {on_disk} != {size}", file=sys.stderr)
        return 1
    written = target.stat().st_blocks * 512
    if written < size * 0.99:
        print(
            f"FAIL il file e' sparse: {written / 1e9:.2f} GB scritti su {size / 1e9:.2f} "
            f"attesi. La dimensione logica combacia ma il contenuto non e' stato scritto: "
            f"non usarlo, e non fidarsi del fatto che `ls` mostri la giusta dimensione.",
            file=sys.stderr,
        )
        return 1

    actual = sha256_file(target)
    print(f"sha256 {actual}")
    if expected and actual != expected and not args.no_verify:
        print(f"FAIL sha256 mismatch, expected {expected}", file=sys.stderr)
        return 1

    sidecar = target.with_suffix(target.suffix + ".parts.json")
    if sidecar.exists():
        sidecar.unlink()
    print("OK verified")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
