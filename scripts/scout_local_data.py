"""Read-only filesystem/data scout. All inventories stay outside the Git checkout.

No imports, resampling, training, label inference, or dataset registration occur.
Optional readers: soundfile, scipy, openpyxl; video headers use ffprobe.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import zipfile

AUDIO = {'.wav', '.flac', '.mp3', '.ogg', '.aif', '.aiff', '.m4a'}
VIDEO = {'.mp4', '.avi', '.mov', '.mkv', '.mpg', '.mpeg'}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def distribution(values):
    values = list(values)
    if not values:
        return None
    return dict(n=len(values), minimum=min(values), median=statistics.median(values),
                maximum=max(values), total=sum(values))


def inspect_file(item):
    path, rel, size, mtime, mtime_ns = item
    ext = path.suffix.lower() or '[none]'
    row = dict(path=rel, bytes=size, extension=ext, modified_utc=mtime)
    payload = None
    signature = b''
    try:
        if size <= 4 * 1024 * 1024:
            payload = path.read_bytes()
            signature = payload[:200]
            row['sha256'] = hashlib.sha256(payload).hexdigest()
        else:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    if not signature:
                        signature = block[:200]
                    h.update(block)
            row['sha256'] = h.hexdigest()
    except Exception as exc:
        row['hash_error'] = f'{type(exc).__name__}: {exc}'
    try:
        if signature.startswith(b'version https://git-lfs.github.com/spec/v1'):
            row['git_lfs_pointer'] = signature.decode('ascii', errors='replace').strip()
        elif ext in AUDIO:
            import soundfile as sf
            info = sf.info(io.BytesIO(payload) if payload is not None else str(path))
            row['audio'] = dict(sample_rate=info.samplerate, channels=info.channels,
                                frames=info.frames, duration_s=info.duration,
                                format=info.format, subtype=info.subtype,
                                endian=info.endian)
        elif ext in VIDEO:
            ffprobe = shutil.which('ffprobe')
            if ffprobe:
                p = subprocess.run([ffprobe, '-v', 'error', '-show_format', '-show_streams',
                                    '-of', 'json', str(path)], capture_output=True,
                                   text=True, timeout=45)
                if p.returncode:
                    raise ValueError(p.stderr.strip()[:1000])
                row['video'] = json.loads(p.stdout)
                row['video'].get('format', {}).pop('filename', None)
            else:
                row['probe_note'] = 'ffprobe unavailable'
        elif ext == '.mat':
            from scipy.io import whosmat
            row['mat_variables'] = [dict(name=n, shape=list(s), matlab_class=c)
                                    for n, s, c in whosmat(str(path))]
        elif ext == '.xlsx':
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True, data_only=False)
            row['workbook'] = []
            for ws in wb:
                preview = list(ws.iter_rows(min_row=1, max_row=min(ws.max_row or 1, 8),
                                           max_col=min(ws.max_column or 1, 20), values_only=True))
                row['workbook'].append(dict(sheet=ws.title, rows=ws.max_row,
                                            columns=ws.max_column, first_rows=preview))
            wb.close()
        elif ext in {'.npy', '.npz'}:
            import numpy as np
            if ext == '.npy':
                arr = np.load(path, mmap_mode='r', allow_pickle=False)
                row['array'] = dict(shape=list(arr.shape), dtype=str(arr.dtype))
                del arr
            else:
                with np.load(path, allow_pickle=False) as data:
                    row['array_members'] = {k: dict(shape=list(data[k].shape), dtype=str(data[k].dtype)) for k in data.files}
        elif ext in {'.zip'}:
            with zipfile.ZipFile(path) as z:
                row['archive'] = dict(members=len(z.infolist()),
                                      uncompressed_bytes=sum(i.file_size for i in z.infolist()))
        elif ext in {'.csv', '.tsv'}:
            with path.open(encoding='utf-8-sig', errors='replace', newline='') as f:
                reader = csv.reader(f, delimiter='\t' if ext == '.tsv' else ',')
                row['table'] = dict(header=next(reader, []), data_rows=sum(1 for _ in reader))
        elif ext in {'.md', '.txt', '.rst', '.m', '.xml', '.json', '.yaml', '.yml', '[none]'} and size < 2_000_000:
            raw = payload if payload is not None else path.read_bytes()
            if b'\x00' not in raw[:4096]:
                row['text_preview'] = raw.decode('utf-8-sig', errors='replace')[:24000]
    except Exception as exc:
        row['probe_error'] = f'{type(exc).__name__}: {exc}'
    try:
        after = path.stat()
        row['stable_during_read'] = after.st_size == size and after.st_mtime_ns == mtime_ns
    except OSError:
        row['stable_during_read'] = False
    return row


def summarize(rows, dirs):
    result = dict(files=len(rows), bytes=sum(r['bytes'] for r in rows), directories=len(dirs),
                  extensions={}, audio={}, folder_totals={}, hash_errors=[], probe_errors=[])
    for ext in sorted({r['extension'] for r in rows}):
        members = [r for r in rows if r['extension'] == ext]
        result['extensions'][ext] = dict(files=len(members), bytes=sum(r['bytes'] for r in members))
    audios = [r for r in rows if 'audio' in r]
    for field in ['sample_rate', 'channels', 'format', 'subtype']:
        result['audio'][field] = dict(Counter(str(r['audio'][field]) for r in audios))
    result['audio']['readable_files'] = len(audios)
    result['audio']['total_audio_files'] = sum(r['extension'] in AUDIO for r in rows)
    result['audio']['metadata_only_files'] = sum(r.get('inspection') == 'filesystem-metadata-only' and r['extension'] in AUDIO for r in rows)
    result['audio']['duration_s'] = distribution(r['audio']['duration_s'] for r in audios)
    result['audio']['frames'] = distribution(r['audio']['frames'] for r in audios)
    hashes = Counter(r['sha256'] for r in rows if 'sha256' in r)
    result['sha256'] = dict(hashed_files=sum(hashes.values()), unique_contents=len(hashes),
                            duplicate_groups=sum(n > 1 for n in hashes.values()),
                            extra_duplicate_copies=sum(n - 1 for n in hashes.values()),
                            files_in_duplicate_groups=sum(n for n in hashes.values() if n > 1))
    for row in rows:
        parts = Path(row['path']).parts
        for i in range(len(parts)):
            folder = '/'.join(parts[:i]) or '.'
            counter = result['folder_totals'].setdefault(folder, dict(files=0, bytes=0))
            counter['files'] += 1
            counter['bytes'] += row['bytes']
        for key in ['hash_error', 'probe_error']:
            if key in row:
                result[key.replace('_error', '_errors')].append(dict(path=row['path'], error=row[key]))
    return result


def scan(root, out, workers, audio_sample_per_folder=0):
    started = datetime.now(timezone.utc).isoformat()
    files, dirs, skipped, errors = [], [], [], []
    pending = [root]
    while pending:
        current = pending.pop()
        dirs.append(current.relative_to(root).as_posix())
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    p = Path(entry.path)
                    if entry.is_symlink() or (entry.is_dir(follow_symlinks=False) and os.path.isjunction(entry.path)):
                        skipped.append(p.relative_to(root).as_posix())
                    elif entry.is_dir(follow_symlinks=False):
                        pending.append(p)
                    elif entry.is_file(follow_symlinks=False):
                        s = entry.stat(follow_symlinks=False)
                        files.append((p, p.relative_to(root).as_posix(), s.st_size,
                                      datetime.fromtimestamp(s.st_mtime, timezone.utc).isoformat(), s.st_mtime_ns))
        except OSError as exc:
            errors.append(str(exc))
    print(f'{root.name}: enumerated {len(files):,} files; {sum(f[2] for f in files):,} bytes', flush=True)
    rows = []
    inspected = files
    if audio_sample_per_folder:
        by_folder = defaultdict(list)
        inspected = []
        for item in files:
            if item[0].suffix.lower() in AUDIO:
                by_folder[item[0].parent].append(item)
            else:
                inspected.append(item)
        for members in by_folder.values():
            members.sort(key=lambda x: x[1])
            # Evenly spaced sorted filenames, including endpoints; deterministic.
            n = min(audio_sample_per_folder, len(members))
            indices = {round(i * (len(members) - 1) / max(n - 1, 1)) for i in range(n)}
            for i, item in enumerate(members):
                if i in indices:
                    inspected.append(item)
                else:
                    rows.append(dict(path=item[1], bytes=item[2], extension=item[0].suffix.lower(),
                                     modified_utc=item[3], inspection='filesystem-metadata-only'))
    print(f'{root.name}: content inspection selected {len(inspected):,} files', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(inspect_file, item) for item in inspected]
        for n, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if n % 5000 == 0:
                print(f'{root.name}: inspected {n:,}/{len(inspected):,}', flush=True)
    rows.sort(key=lambda r: r['path'])
    result = dict(root=str(root), started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                  directories=sorted(dirs), skipped_links=skipped, walk_errors=errors,
                  audio_sample_per_folder=audio_sample_per_folder,
                  summary=summarize(rows, dirs), files=rows)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=True, default=str), encoding='utf-8')
    print(f'{root.name}: saved {out.name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--audio-sample-per-folder', type=int, default=0,
                        help='0 inspects all audio; positive values sample this many files per leaf folder')
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.is_relative_to(checkout):
        parser.error('Private inventories must be written outside the catalog Git checkout.')
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.audio_sample_per_folder < 0:
        parser.error('--audio-sample-per-folder must be non-negative')
    roots = [p.resolve(strict=True) for p in args.root]
    if any(not p.is_dir() for p in roots):
        parser.error('Every --root must be a directory')
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in roots):
        parser.error('Output and scanned roots must be disjoint')
    if len({p.name.casefold() for p in roots}) != len(roots):
        parser.error('Root directory names must be distinct')
    output.mkdir(parents=True, exist_ok=True)
    for root in roots:
        scan(root, output / (root.name + '.json'), args.workers, args.audio_sample_per_folder)


if __name__ == '__main__':
    main()
