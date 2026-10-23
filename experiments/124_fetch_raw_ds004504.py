"""Fetch the UNPROCESSED ds004504 recordings from OpenNeuro.

Why this is needed. The copy of ds004504 in `datasets/openneuro_ds004504/` contains only the
`derivatives/` tree, which its authors had already band-passed, run through Artifact Subspace
Reconstruction, and ICA-cleaned with eye and jaw components removed. Every other cohort in this
project arrives raw. So the one cohort that carries the paper's external dementia result -- and the
site where the artefact claim is made -- is also the only cohort whose muscle has already been
partly removed by someone else.

That asymmetry cannot be argued away; it has to be removed. OpenNeuro hosts the original
`sub-XXX/eeg/*.set` files alongside the derivatives, so the raw data is one download away.

The files land in `datasets/openneuro_ds004504_raw/`, a SEPARATE directory. The existing
derivative-based dataset is left untouched so that every number already computed remains
reproducible and the two can be compared directly -- which is itself an experiment worth running,
since it measures what the authors' cleaning did to the concepts.

Robustness matters here: a first attempt failed with `ContentTooShortError` after 1.2 MB of 24 MB.
Each file is therefore downloaded to a temporary name, checked against the size the API reports,
and only then moved into place. Re-running skips files already complete, so an interrupted
download resumes at file granularity.

    python experiments/124_fetch_raw_ds004504.py            # fetch
    python experiments/124_fetch_raw_ds004504.py --check    # verify only, no download
Out: datasets/openneuro_ds004504_raw/  (~2.1 GB, 88 subjects)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEST = ROOT / "datasets" / "openneuro_ds004504_raw"
DATASET, TAG = "ds004504", "1.0.9"
GRAPHQL = "https://openneuro.org/crn/graphql"
RETRIES, TIMEOUT = 6, 120
#: Abort and resume a transfer that sits below 10 kB/s for 45 s. Observed healthy rate on this
#: link is ~175 kB/s, so the floor is ~17x below normal and will not fire on a merely slow link.
STALL_BYTES_PER_S, STALL_SECONDS = 10_240, 45
#: Hard ceiling per attempt, in case curl itself hangs. 24 MB at the 10 kB/s floor is ~40 min;
#: 15 min forces a resume well before that.
CURL_TIMEOUT = 900


def gq(query: str):
    req = urllib.request.Request(GRAPHQL, data=json.dumps({"query": query}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as fh:
        return json.loads(fh.read().decode())


def tree(tree_id: str | None = None):
    arg = f'(tree:"{tree_id}")' if tree_id else ""
    q = (f'query{{snapshot(datasetId:"{DATASET}",tag:"{TAG}")'
         f'{{files{arg}{{filename id size directory urls}}}}}}')
    return gq(q)["data"]["snapshot"]["files"]


def fetch(url: str, dest: Path, expect: int) -> bool:
    """Download with size verification. Returns True if the file is complete on disk.

    Transferred by `curl` rather than `urllib`. The urllib version stalled indefinitely on
    ds004504: sub-041 stopped at exactly 5 MiB and `read()` simply never returned again, while the
    connection stayed open so the socket timeout never fired. A stalled transfer that neither
    progresses nor raises is the worst failure mode for an unattended job -- it looks identical to
    a slow one.

    curl solves it directly: `--speed-limit`/`--speed-time` abort a transfer that drops below a
    floor for a sustained period, and `-C -` resumes a partial file rather than restarting it, so
    an abort costs only the bytes not yet received.
    """
    if dest.exists() and dest.stat().st_size == expect:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, RETRIES + 1):
        cmd = ["curl", "-sS", "-L", "--fail",
               "--speed-limit", str(STALL_BYTES_PER_S), "--speed-time", str(STALL_SECONDS),
               "--connect-timeout", "30", "--retry", "2", "--retry-delay", "3",
               "-o", str(tmp), url]
        if tmp.exists() and tmp.stat().st_size:
            cmd.insert(1, "-C")            # resume from where the last attempt stopped
            cmd.insert(2, "-")
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=CURL_TIMEOUT)
        except subprocess.TimeoutExpired:
            res = None
        got = tmp.stat().st_size if tmp.exists() else 0
        if res is not None and res.returncode == 0 and (not expect or got == expect):
            # On Windows the handle curl held on the .part file is not always released by the
            # time subprocess.run returns -- an on-access virus scanner commonly keeps it open for
            # a moment -- and the move fails with WinError 32. Retrying briefly is correct here:
            # the bytes are already verified, only the rename is contended.
            for delay in (0.0, 0.3, 1.0, 3.0):
                time.sleep(delay)
                try:
                    shutil.move(str(tmp), str(dest))
                    return True
                except PermissionError:
                    continue
            print(f"    could not rename {tmp.name} (file locked); will retry the file")
        why = ("hard timeout" if res is None else
               f"curl {res.returncode}: {(res.stderr or '').strip()[:110]}"
               if res.returncode else f"size {got:,} != {expect:,}")
        if attempt == RETRIES:
            print(f"    FAILED after {RETRIES} attempts ({why})")
            return False
        wait = 2 ** attempt
        print(f"    attempt {attempt} stalled/failed ({why}); resuming in {wait}s "
              f"[{got:,}/{expect:,} B]")
        time.sleep(wait)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify what is present, download nothing")
    ap.add_argument("--limit", type=int, default=0, help="stop after N subjects (for testing)")
    args = ap.parse_args()

    print(f"{DATASET} snapshot {TAG} -> {DEST}")
    top = tree()
    subs = sorted([f for f in top if f["directory"] and f["filename"].startswith("sub-")],
                  key=lambda f: f["filename"])
    root_files = [f for f in top if not f["directory"]]
    print(f"{len(subs)} subject directories, {len(root_files)} root files")
    if args.limit:
        subs = subs[:args.limit]

    DEST.mkdir(parents=True, exist_ok=True)
    ok = failed = skipped = 0
    total_bytes = 0

    # dataset-level files (participants.tsv carries the MMSE the audit depends on)
    for f in root_files:
        d = DEST / f["filename"]
        if args.check:
            print(f"  {'OK ' if d.exists() else 'MISSING'} {f['filename']}")
            continue
        if fetch(f["urls"][0], d, f["size"]):
            ok += 1
            total_bytes += f["size"]
        else:
            failed += 1

    for i, s in enumerate(subs, 1):
        name = s["filename"]
        eeg_dirs = [x for x in tree(s["id"]) if x["directory"] and x["filename"] == "eeg"]
        if not eeg_dirs:
            print(f"  [{i}/{len(subs)}] {name}: no eeg/ directory")
            continue
        files = tree(eeg_dirs[0]["id"])
        for f in files:
            d = DEST / name / "eeg" / f["filename"]
            if d.exists() and d.stat().st_size == f["size"]:
                skipped += 1
                continue
            if args.check:
                print(f"  MISSING {name}/eeg/{f['filename']} ({f['size']:,} B)")
                failed += 1
                continue
            if fetch(f["urls"][0], d, f["size"]):
                ok += 1
                total_bytes += f["size"]
            else:
                failed += 1
        if i % 10 == 0 or i == len(subs):
            print(f"  [{i}/{len(subs)}] {name} | ok {ok} skipped {skipped} failed {failed} "
                  f"| {total_bytes/1e9:.2f} GB", flush=True)

    print(f"\ndownloaded {ok}, already present {skipped}, failed {failed} "
          f"({total_bytes/1e9:.2f} GB fetched)")
    if failed:
        print("Re-run to retry the failures; completed files are skipped.")
        return 1
    print(f"\nRaw data is at {DEST}. The derivative-based copy in "
          f"datasets/openneuro_ds004504/ is untouched,\nso the two can be compared -- which "
          f"measures what the authors' ASR and ICA did to the concepts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
