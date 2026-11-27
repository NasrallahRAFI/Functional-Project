"""Collect factual registry, project-stage, and ZIP evidence for a local scout.

Reads metadata and existing manifests only. Does not score held-out data, fit
statistics, modify source files, or establish new corpus/split identities.
"""
import argparse
from collections import Counter
import csv
import io
import json
import os
from pathlib import Path
import zipfile

from scout_local_data import digest


def iter_files(root):
    pending = [root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_symlink() or (entry.is_dir(follow_symlinks=False) and os.path.isjunction(entry.path)):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    yield Path(entry.path), entry.stat(follow_symlinks=False)


def table_profile(reader, has_header=True):
    header = next(reader, []) if has_header else []
    count, preview = 0, []
    splits = Counter()
    for row in reader:
        count += 1
        if len(preview) < 5:
            preview.append(row)
        if 'split' in header and len(row) > header.index('split'):
            splits[row[header.index('split')]] += 1
    return dict(header=header, has_header=has_header, data_rows=count, first_rows=preview, split_counts=dict(splits))


def array_header(stream):
    import numpy as np
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f'Unsupported NPY header version: {version}')
    return dict(shape=list(shape), dtype=str(dtype), fortran=fortran)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--downloads', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    project = a.project.resolve(strict=True)
    downloads = a.downloads.resolve(strict=True)
    output = a.output.resolve()
    checkout = Path(__file__).resolve().parents[1]
    if output.is_relative_to(checkout) or any(output.is_relative_to(project / 'storage' / n) for n in ['sources', 'corpora', 'splits', 'caches', 'derived']):
        p.error('Output must be outside the checkout and input storage trees')
    output.mkdir(parents=True, exist_ok=True)
    result = dict(registry={}, source_inventory_checks=[], stages={}, archives=[], supplied_metadata=[])
    for path in sorted((checkout / 'registry').rglob('*.json')):
        if 'local' not in path.relative_to(checkout / 'registry').parts:
            result['registry'][path.relative_to(checkout).as_posix()] = json.loads(path.read_text(encoding='utf-8-sig'))
    for name, entry in result['registry'].items():
        if not name.startswith('registry/sources/'):
            continue
        manifest = project / entry['inventory']
        result['source_inventory_checks'].append(dict(id=entry['id'],
            expected_sha256=entry['inventory_sha256'], observed_sha256=digest(manifest)))
    for stage in ['corpora', 'splits', 'caches', 'derived']:
        base = project / 'storage' / stage
        for root in sorted(base.iterdir()):
            if not root.is_dir() or root.is_junction() or root.is_symlink():
                continue
            entries = []
            for path, stat in iter_files(root):
                row = dict(path=path.relative_to(root).as_posix(), bytes=stat.st_size,
                           extension=path.suffix.lower() or '[none]')
                try:
                    if path.suffix == '.npy':
                        with path.open('rb') as f:
                            # Header only; never load feature arrays or held-out signals.
                            row['array_header'] = array_header(f)
                    elif path.suffix == '.npz':
                        row['array_members'] = {}
                        with zipfile.ZipFile(path) as z:
                            for name in z.namelist():
                                if name.endswith('.npy'):
                                    with z.open(name) as stream:
                                        row['array_members'][name] = array_header(stream)
                    elif path.suffix == '.csv':
                        with path.open(encoding='utf-8-sig', newline='') as f:
                            reader = csv.DictReader(f)
                            columns = reader.fieldnames
                            counts = {k: Counter() for k in ['source', 'label', 'label_name', 'split', 'kept', 'exclusion_reason'] if k in (columns or [])}
                            n = 0
                            for r in reader:
                                n += 1
                                for k in counts:
                                    counts[k][r[k]] += 1
                        row['table'] = dict(columns=columns, rows=n, value_counts={k:dict(v) for k,v in counts.items()})
                    elif path.suffix == '.json' and stat.st_size < 3_000_000:
                        row['json'] = json.loads(path.read_text(encoding='utf-8-sig'))
                    elif path.suffix in {'.txt', '.md'} and stat.st_size < 200_000:
                        row['text'] = path.read_text(encoding='utf-8-sig', errors='replace')[:24000]
                except Exception as exc:
                    row['probe_error'] = f'{type(exc).__name__}: {exc}'
                entries.append(row)
            result['stages'][root.relative_to(project).as_posix()] = dict(files=len(entries), bytes=sum(r['bytes'] for r in entries), entries=entries)
            print(f'Project stage: {root.name}, {len(entries):,} files', flush=True)
    # Archives are inspected through central-directory records; no extraction.
    drone = downloads / 'Drone-20261125T142337Z-1-001' / 'Drone'
    archive_paths = [drone / 'itu-aris-lab-acoustic-drone-dataset-v1.0.0.zip']
    archive_paths.extend(sorted((drone / 'FSD50K').glob('*.zip')))
    for path in archive_paths:
        item = dict(path=str(path), bytes=path.stat().st_size, members=[], tables=[], text=[])
        with zipfile.ZipFile(path) as z:
            for member in z.infolist():
                item['members'].append(dict(path=member.filename, bytes=member.file_size,
                    compressed_bytes=member.compress_size, crc32=f'{member.CRC:08x}', directory=member.is_dir()))
                if member.is_dir():
                    continue
                ext = Path(member.filename).suffix.lower()
                if ext in {'.csv', '.tsv'}:
                    with z.open(member) as stream:
                        reader = csv.reader(io.TextIOWrapper(stream, encoding='utf-8-sig', errors='replace'), delimiter='\t' if ext == '.tsv' else ',')
                        item['tables'].append(dict(path=member.filename, **table_profile(reader, has_header=Path(member.filename).name != 'vocabulary.csv')))
                elif member.file_size < 2_000_000 and (ext in {'.json', '.md', '.txt'} or Path(member.filename).name.lower().startswith(('license', 'readme'))):
                    item['text'].append(dict(path=member.filename, content=z.read(member).decode('utf-8-sig', errors='replace')[:24000]))
        item['uncompressed_bytes'] = sum(m['bytes'] for m in item['members'])
        result['archives'].append(item)
        print(f'Archive: {path.name}, {len(item["members"]):,} members', flush=True)
    for root in [drone / 'ITU_ARIS_Acoustic_Drone_Dataset', drone / 'ITU_ARIS_Zenodo_22682339' / 'itu-aris-lab-acoustic-drone-dataset']:
        for name in ['metadata.json', 'groups.json', 'splits/split_interval_disjoint.json', 'splits/split_temporal.json']:
            path = root / name
            if path.exists():
                data = json.loads(path.read_text(encoding='utf-8-sig'))
                result['supplied_metadata'].append(dict(path=str(path), content=data))
    (output / 'supporting_evidence.json').write_text(json.dumps(result, indent=2, ensure_ascii=True, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
