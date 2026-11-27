"""Render a factual local scout PDF and complete searchable CSV companions.

Consumes scout_local_data.py inventories plus collect_scout_evidence.py output.
Private paths and file inventories are never written to the catalog checkout.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from html import escape
import itertools
import json
from pathlib import Path
import re

from scout_local_data import AUDIO, distribution, summarize


NOTES = {
 'Al-Emadi (registered)': [
  'Registered source al-emadi-r001; used by corpus-001 (V1/V2/V3) and corpus-002 (V4). Only the binary audio membership is selected by those corpora.',
  'The original download preserves Binary_Drone_Audio and Multi_Class_Drone_Audio. Their audio content overlaps; these are two task layouts, not twice as many independent recordings.',
  'The local README describes indoor drone-propeller recordings, added random noise, ESC-derived unknown sounds, Speech Commands-related white noise, and author-created silence. These acquisition descriptions are source statements, not independently verified observations.',
  'The local README/LICENSE describe educational and non-commercial academic research terms, restricted application categories, and separate third-party material terms. This is a description of the local documents, not a legal assessment.',
  'The existing corpus-001 record has 11,704 raw binary rows and 10,623 retained rows. The V4 preparation report records 1,079 exact-zero duplicate removals and two clips below its minimum-length requirement. No cleaning was rerun for this scout.'
 ],
 'Drone Detection Thesis (registered)': [
  'Registered source thesis-r001. The 90 native audio recordings are included in V4 corpus-002; the video and MATLAB labels are imported source material, not audio-corpus membership.',
  'Native audio filenames encode BACKGROUND, DRONE, or HELICOPTER. Each category contains 30 recordings. V4 stores 900 non-overlapping one-second derivatives and records 890 retained thesis segments after duplicate exclusions.',
  'The local README describes 650 labelled videos: 365 infrared and 285 visible, with airplane, bird, drone and helicopter labels. The filesystem also contains two root-level MP4 examples, so the MP4 file count is 652.',
  'MATLAB label files are Computer Vision groundTruth/MCOS objects according to the local README. The basic SciPy metadata reader did not decode them; this does not establish label corruption or empty annotations.',
  'The local LICENSE identifies CC0 1.0 Universal. Video_dataset_description.xlsx supplies sensor, class, number, distance bin, bin index, and drone/internet-source fields.'
 ],
 'Salford DroneNoise (registered)': [
  'Registered source salford-r001 is imported locally. It is not selected by corpus-001; corpus-002 explicitly lists it as excluded. Registered presence is distinct from training membership.',
  'The source contains 174 WAVs and DNM_SQM_FLYO_AESCT.xlsx, without a local README or license file. The registered URL identifies the source record; exact downloaded publication revision is not established.',
  'The existing local comparison describes 165 event-style recordings and nine calibration files. The workbook contains flyover metadata and microphone acoustic metrics. File counts do not establish independent flight counts or segment-level binary labels.'
 ],
 'drone_A': ['A/train, A/valid and A/test each contain mic1 and mic2 folders. These are supplied folder partitions; this scout does not certify independence between them.',
             'Names contain underscore-separated tokens and numeric snr= values. There is no local README or separate label table in this tree. Meanings of abbreviated filename tokens are not asserted.',
             'All file sizes and paths are inventoried. Audio headers and hashes use 64 deterministic, evenly spaced sorted filenames per leaf folder; durations and format frequencies describe that sample only.'],
 'drone_B': ['The tree has B/train, B/valid and B/test, each with mic1 and mic2. The filename structure resembles A, but names alone do not establish the relationship between the collections.',
             'No local README or separate label table was found. Header/hash evidence is sampled at 64 files per leaf folder; the filesystem inventory is complete.'],
 'drone_C': ['The tree has C/train, C/valid and C/test, each with mic1 and mic2. These are source folder names, not project split-001 or split-002.',
             'No local README or separate label table was found. Header/hash evidence is sampled at 64 files per leaf folder; the filesystem inventory is complete.'],
 'MLSP 2022 real data': ['The real_data tree is arranged in recording/session-named folders, containing many short multichannel audio files. Names include timestamp-like strings and session identifiers; their semantics are not inferred without source documentation.',
                       'All file sizes and paths are inventoried. Audio metadata uses up to 64 deterministic samples per leaf folder. Aggregate sample duration is not the duration of the full collection.'],
 'MicArrayData30s': ['The main tree contains session/chunk folders with audio and DOATruePos.csv files. CSV fields observed locally are time, X, Y, Z, ele and azi; an inspected table starts at time 0.0 with 0.01 increments.',
                    'The __MACOSX tree and .DS_Store/._ files are operating-system sidecars. A ._ file with an audio suffix is not counted as successfully decoded audio. Geometry units, coordinate frame and synchronization accuracy are not independently established.'],
 'MUSAN': ['The local README describes a corpus divided into music, speech and noise, with subdirectories by source. It states approximately 109 hours; measured header durations in this report describe the actual local files.',
           'LICENSE and ANNOTATIONS documents provide item attribution and category metadata. The README describes Creative Commons and US public-domain source material. The scout records local document presence and content, not a legal interpretation.'],
 'Thesis download copy': ['This is a separate download-tree location corresponding to the registered thesis source. The exact-content overlap section reports SHA-256 matches; a second filesystem copy does not establish additional recordings.'],
 'DroneAudioSet download': ['The local README declares drone-only and drone-with-source configurations, numbered train shards, and MIT metadata. Empty directories, cache metadata, lock files and incomplete markers are inventoried separately from usable payload.',
                           'Documentation claims are not treated as proof that the corresponding audio or data shards were downloaded.'],
 'FSD50K metadata archives': ['Only FSD50K.doc.zip and FSD50K.ground_truth.zip are present in this local subtree. Their central directories contain documentation and CSV ground truth, not audio payloads.',
                            'The vocabulary CSV has no header and contains 200 rows. dev.csv contains 40,966 rows (36,796 train and 4,170 val); eval.csv contains 10,231 rows. These are metadata records, not locally inventoried audio files.'],
 'ITU ARIS partial download': ['This folder has the same dataset README title as the fuller ITU copies, but the audio payload is only a partial local subset. Counts are measured below; README class totals describe the source, not this folder.',
                              'Cache and incomplete-download markers are included in the filesystem inventory. Dataset declarations do not establish that all listed group or split files are present.'],
 'ITU ARIS snapshot': ['Local metadata declares version 1.0.0, 4,491 drone and 1,000 background segments at 22,050 Hz mono PCM16, one second per clip.',
                       'The supplied groups.json describes 50% segment overlap, 35 source drone intervals and background blocks. These are metadata declarations, not independently reconstructed source intervals.',
                       'Two supplied assignment files exist: interval-disjoint (3,844 train / 823 val / 824 test) and temporal (3,577 train / 816 val / 1,098 test). Neither is a registered project split.',
                       'The local metadata describes a single 3.12-hour outdoor source session and CC-BY-4.0. The original continuous session is not established as locally present by these segment files.'],
 'ITU ARIS extracted archive': ['This is another extracted version 1.0.0 tree with audio, metadata, groups and two supplied split files. Exact file-content matches with the snapshot and ZIP are recorded separately.',
                               'The duplicate local locations are not independent sessions. Local metadata describes one-second segments, 50% overlap, and CC-BY-4.0.'],
 'Wang drone signatures': ['This is a React audio-visualization project containing raw audio under public, drone images, mel/MFCC plot images, and src/data/drones.json.',
                          'The README describes 32 drones. The actual manifest item count, fields and referenced local assets are inventoried; no recording-condition or calibrated-distance meaning is inferred from numeric name suffixes.',
                          'A website source tree is ancillary material in this scout; no website files were edited. The project catalog also has an external WSU/Wang reference prototype, which is not a registered corpus source.']
}


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=True) if isinstance(v, (dict, list)) else v for k, v in r.items()})


def load_datasets(base):
    datasets, roots = [], []
    for name, folder in [('Al-Emadi (registered)', 'project-al-emadi'), ('Drone Detection Thesis (registered)', 'project-thesis'), ('Salford DroneNoise (registered)', 'project-salford')]:
        d = json.loads((base / folder / 'original.json').read_text())
        d.update(name=name, category='registered-source')
        roots.append(d)
        datasets.append(d)
    names = {'drone_A':'drone_A', 'drone_B':'drone_B', 'drone_C':'drone_C', 'Drone-detection-dataset-master':'Thesis download copy', 'MicArrayData30s':'MicArrayData30s', 'MLSP_2022_Real_Data':'MLSP 2022 real data', 'musan':'MUSAN'}
    for file, name in names.items():
        d = json.loads((base / 'inventories' / (file + '.json')).read_text())
        d.update(name=name, category='download-location')
        roots.append(d)
        datasets.append(d)
    drone = json.loads((base / 'inventories' / 'Drone-20261125T142337Z-1-001.json').read_text())
    drone.update(name='Drone download collection', category='download-location')
    roots.append(drone)
    names = {'DroneAudioSet':'DroneAudioSet download', 'FSD50K':'FSD50K metadata archives', 'ITU ARIS Lab Acoustic Drone Dataset':'ITU ARIS partial download', 'ITU_ARIS_Acoustic_Drone_Dataset':'ITU ARIS snapshot', 'ITU_ARIS_Zenodo_22682339':'ITU ARIS extracted archive', 'Wang drone signatures':'Wang drone signatures'}
    for folder, name in names.items():
        prefix = 'Drone/' + folder + '/'
        rows = [{**r, 'path':r['path'][len(prefix):]} for r in drone['files'] if r['path'].startswith(prefix)]
        dirs = [p[len(prefix):] or '.' for p in drone['directories'] if p.startswith(prefix)]
        datasets.append(dict(name=name, root=str(Path(drone['root']) / 'Drone' / folder), category='download-subcollection', files=rows,
                             summary=summarize(rows, dirs), directories=dirs, audio_sample_per_folder=0,
                             started_utc=drone['started_utc'], finished_utc=drone['finished_utc'], walk_errors=[], skipped_links=[]))
    return datasets, roots, drone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    base = args.evidence.resolve(strict=True)
    project = args.project.resolve(strict=True)
    checkout = Path(__file__).resolve().parents[1]
    if base.is_relative_to(checkout):
        parser.error('Private reports must stay outside the catalog checkout')
    datasets, roots, drone = load_datasets(base)
    support = json.loads((base / 'supporting_evidence.json').read_text())
    # Complete per-file and per-folder companions, including metadata-only rows.
    def file_rows():
        for d in roots:
            for r in d['files']:
                yield dict(root=d['root'], collection=d['name'], **r, **r.get('audio', {}))
    write_csv(base / 'file_inventory.csv', ['collection','root','path','bytes','extension','modified_utc','inspection','sha256','stable_during_read','format','subtype','sample_rate','channels','frames','duration_s','probe_error','hash_error','git_lfs_pointer'], file_rows())
    def folder_rows():
        for d in roots:
            for folder in d['directories']:
                yield dict(collection=d['name'], root=d['root'], path=folder,
                           **d['summary']['folder_totals'].get(folder, dict(files=0, bytes=0)))
    write_csv(base / 'folder_inventory.csv', ['collection','root','path','files','bytes'], folder_rows())
    hashes = defaultdict(list)
    for d in roots:
        for r in d['files']:
            if 'sha256' in r:
                hashes[r['sha256']].append(dict(collection=d['name'], root=d['root'], path=r['path'], bytes=r['bytes']))
    write_csv(base / 'exact_duplicates.csv', ['sha256','copies','collection','root','path','bytes'],
              (dict(sha256=h, copies=len(members), **r) for h, members in hashes.items() if len(members) > 1 for r in members))
    write_csv(base / 'project_stage_inventory.csv', ['stage','path','bytes','extension','array_header','array_members','table','probe_error'],
              (dict(stage=k, **r) for k,v in support['stages'].items() for r in v['entries']))
    write_csv(base / 'archive_inventory.csv', ['archive','path','bytes','compressed_bytes','crc32','directory'],
              (dict(archive=a['path'], **r) for a in support['archives'] for r in a['members']))
    audio_rows = []
    for d in datasets:
        groups = defaultdict(list)
        for r in d['files']:
            if 'audio' in r:
                a = r['audio']
                groups[(a['format'],a['subtype'],a['sample_rate'],a['channels'])].append(a['duration_s'])
        for key, durations in sorted(groups.items()):
            audio_rows.append(dict(collection=d['name'], scope='sampled' if d.get('audio_sample_per_folder') else 'all-files', format=key[0], subtype=key[1], sample_rate=key[2], channels=key[3], **distribution(durations)))
    write_csv(base / 'audio_profiles.csv', ['collection','scope','format','subtype','sample_rate','channels','n','minimum','median','maximum','total'], audio_rows)
    write_csv(base / 'metadata_inventory.csv', ['collection','root','path','table','workbook','mat_variables','git_lfs_pointer','probe_error'],
              (dict(collection=d['name'],root=d['root'],**r) for d in roots for r in d['files'] if any(k in r for k in ['table','workbook','mat_variables','git_lfs_pointer','probe_error']) and not Path(r['path']).name.startswith('._')))
    folder_audio=defaultdict(list)
    for d in roots:
        for r in d['files']:
            if 'audio' in r:
                a=r['audio']
                folder_audio[(d['name'],str(Path(r['path']).parent),a['format'],a['subtype'],a['sample_rate'],a['channels'])].append(a['duration_s'])
    write_csv(base/'audio_by_folder.csv',['collection','folder','format','subtype','sample_rate','channels','n','minimum','median','maximum','total'],
              (dict(collection=k[0],folder=k[1],format=k[2],subtype=k[3],sample_rate=k[4],channels=k[5],**distribution(v)) for k,v in sorted(folder_audio.items())))
    write_csv(base/'filename_metadata.csv',['collection','first_three_tokens','files'],
              (dict(collection=d['name'],first_three_tokens=k,files=v) for d in datasets if d['name'] in {'drone_A','drone_B','drone_C'} for k,v in sorted(Counter('_'.join(Path(r['path']).stem.split('_')[:3]) for r in d['files']).items())))
    # This aggregate is deliberately path-free and contains no individual recordings.
    aggregate = dict(schema_version=1, title='Factual local data scout', snapshot_date='2026-11-27',
        scope='Three registered sources and eight download roots; no recommendations or new study membership.',
        count_semantics='Filesystem files and logical bytes, including copies and sidecars. Audio durations are sums of files, not independent acquisition time. Sampled collections explicitly identify header/hash coverage.',
        collections=[dict(name=d['name'],category=d['category'],files=d['summary']['files'],bytes=d['summary']['bytes'],
                          extensions=d['summary']['extensions'],audio=d['summary']['audio'],
                          header_sampling_per_leaf=d.get('audio_sample_per_folder',0),
                          hash_coverage=d['summary']['sha256'], probe_error_count=len(d['summary']['probe_errors'])) for d in datasets])
    (base / 'aggregate_summary.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, LongTable, TableStyle, PageBreak, KeepTogether
    fontroot = Path('C:/Windows/Fonts')
    pdfmetrics.registerFont(TTFont('Scout', str(fontroot / 'segoeui.ttf')))
    pdfmetrics.registerFont(TTFont('ScoutBold', str(fontroot / 'segoeuib.ttf')))
    pdfmetrics.registerFontFamily('Scout', normal='Scout', bold='ScoutBold')
    ink, muted, teal = colors.HexColor('#163044'), colors.HexColor('#566875'), colors.HexColor('#007D80')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ScoutBody', fontName='Scout', fontSize=8.6, leading=12, textColor=ink, spaceAfter=7))
    styles.add(ParagraphStyle(name='ScoutSmall', parent=styles['ScoutBody'], fontSize=7.4, leading=10, spaceAfter=3))
    styles.add(ParagraphStyle(name='ScoutCell', parent=styles['ScoutSmall'], fontSize=7, leading=9, splitLongWords=True))
    styles.add(ParagraphStyle(name='ScoutH1', fontName='ScoutBold', fontSize=21, leading=26, textColor=ink, spaceAfter=15))
    styles.add(ParagraphStyle(name='ScoutH2', fontName='ScoutBold', fontSize=12, leading=16, textColor=teal, spaceBefore=13, spaceAfter=7))
    styles.add(ParagraphStyle(name='ScoutTitle', parent=styles['ScoutH1'], fontSize=33, leading=38, spaceAfter=20))
    story, text_lines = [], []
    width = A4[0] - 86
    def para(text, style='ScoutBody'):
        text = str(text).replace('\u2011', '-').replace('\u2013','-').replace('\u2014','-')
        story.append(Paragraph(escape(text).replace('\n','<br/>'), styles[style]))
        text_lines.append(text)
    def heading(text):
        para(text, 'ScoutH2')
    def table(headers, rows, weights=None):
        if not rows:
            return
        data = [[Paragraph(escape(str(x)), styles['ScoutCell']) for x in row] for row in [headers] + rows]
        weights = weights or [1] * len(headers)
        t = LongTable(data, colWidths=[width * w / sum(weights) for w in weights], repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DCEBEC')),('VALIGN',(0,0),(-1,-1),'TOP'),
                               ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
                               ('LINEBELOW',(0,0),(-1,0),0.6,teal),('LINEBELOW',(0,1),(-1,-1),0.25,colors.HexColor('#D7E0E4'))]))
        story.extend([t, Spacer(1,8)])
        text_lines.append(' | '.join(map(str, headers)))
        text_lines.extend(' | '.join(map(str,row)) for row in rows)
    def size(n):
        return f'{n:,} B / {n / 2**30:.3f} GiB'
    def seconds(n):
        return f'{n:.6f}'
    def page(title):
        story.append(PageBreak())
        para(title, 'ScoutH1')

    para('LOCAL DATA / FACTUAL INVENTORY', 'ScoutH2')
    para('Project data\n& download scout', 'ScoutTitle')
    para('Snapshot: 27 November 2026 | ML Course Project')
    para('Folder structures, file counts, logical sizes, detected formats, metadata, labels as supplied, source status, exact-copy evidence and observable limitations.')
    para('This is a descriptive scout. It does not assign new data to a study, propose uses, recommend changes, or alter corpus membership, splits, sources or evaluation locks.')
    total_files = sum(d['summary']['files'] for d in roots)
    total_bytes = sum(d['summary']['bytes'] for d in roots)
    table(['Scope','Measured coverage'], [
      ['Registered original-source roots', '3'], ['Download roots','8'],
      ['Filesystem files across those roots', f'{total_files:,}'], ['Logical bytes across those roots',size(total_bytes)],
      ['Prepared project-stage roots',str(len(support['stages']))], ['Reader scope','Complete file inventory; full registered-source inspection; sampled headers/hashes for A/B/C and MLSP']
    ],[1,2])
    para('Totals count each local copy and sidecar. They are not a deduplicated data volume or a count of independent recordings. ZIP members are reported separately and are not added to physical-file totals.')
    heading('Reading the evidence')
    para('Measured: filesystem metadata, file hashes, audio/video headers and parsed table/header structure. Source-stated: README, license and supplied metadata declarations. Project-recorded: registry and existing preparation evidence. Unknown: anything not established by those observations. These distinctions apply throughout.')

    page('Scope and project membership')
    table(['Collection location','Files','Logical size','Status'], [[d['name'],f"{d['summary']['files']:,}",f"{d['summary']['bytes']/2**30:.3f} GiB",'Registered source' if d['category']=='registered-source' else 'Download location'] for d in roots],[2.5,0.8,1,1.2])
    heading('Registered study dependencies')
    table(['Study','Corpus / split','Source membership recorded in catalog'], [
      ['V1 historical','corpus-001 / split-001','Al-Emadi binary'],['V2 frozen baseline','corpus-001 / split-001','Al-Emadi binary'],
      ['V3 active experimental','corpus-001 / split-001','Al-Emadi binary'],['V4 experimental','corpus-002 / split-002','Al-Emadi binary + thesis audio; Salford explicitly excluded']],[1,1.5,2.5])
    para('The remaining download collections do not appear as registered source identities or membership sources in corpus-001/corpus-002. That statement describes the inspected registry; it does not prove they were never mentioned or consulted elsewhere.')
    heading('Existing corpus and split counts')
    rows = []
    for stage,v in support['stages'].items():
        if stage.startswith('storage/splits/'):
            for r in v['entries']:
                if 'table' in r:
                    rows.append([stage.split('/')[-1],r['path'],f"{r['table']['rows']:,}",json.dumps(r['table']['value_counts'].get('label',{}))])
    table(['Identity','Manifest','Rows','Recorded labels'], rows,[1,1,0.7,2])
    para('corpus-001: 11,704 raw binary rows; 10,623 retained. corpus-002: 11,794 native source files, 12,604 manifest rows after thesis segmentation, 11,513 retained. These refer to different stages and must not be added as independent material.')

    page('Collection index')
    para('The numbered profiles below describe every registered raw source and each distinct download collection. The document outline links to these profiles. Per-file paths and complete folder trees are in the CSV companions.')
    table(['Profile','Collection','Files','Decoded audio headers','Coverage'],[[i,d['name'],f"{d['summary']['files']:,}",f"{d['summary']['audio']['readable_files']:,}",'sampled' if d.get('audio_sample_per_folder') else 'all files attempted'] for i,d in enumerate(datasets,1)],[0.5,2.3,0.8,1.1,1])
    para('Zero decoded audio headers may reflect metadata-only downloads. Unreadable audio-suffix macOS sidecars are documented in the affected profile. Coverage does not mean full signal-quality validation.')

    for index,d in enumerate(datasets,1):
        page(f'{index}. {d["name"]}')
        para(d['root'], 'ScoutSmall')
        s = d['summary']
        sampled = bool(d.get('audio_sample_per_folder'))
        para(f"{s['files']:,} filesystem files | {size(s['bytes'])} | {len(d['directories']):,} directories (including root where recorded)")
        sizes=distribution(r['bytes'] for r in d['files'])
        if sizes:
            para(f"File-size range: {sizes['minimum']:,} to {sizes['maximum']:,} bytes; median {sizes['median']:,.0f} bytes.", 'ScoutSmall')
        for note in NOTES.get(d['name'],[]):
            para(note)
        heading('File formats and inspection coverage')
        table(['Extension','Files','Logical bytes'],[[k,f"{v['files']:,}",f"{v['bytes']:,}"] for k,v in s['extensions'].items()],[1,1,1.6])
        audio_suffix = sum(r['extension'] in AUDIO for r in d['files'])
        para(f"Audio-suffix files: {audio_suffix:,}. Successfully read audio headers: {s['audio']['readable_files']:,}. Hashed files: {s['sha256']['hashed_files']:,}/{s['files']:,}. Header/hash scope: {'sample of up to 64 audio files per leaf folder, plus all non-audio files' if sampled else 'all inventoried files attempted'}.")
        profiles = [r for r in audio_rows if r['collection']==d['name']]
        table(['Detected container / encoding','Hz / channels','Read files','Duration min / median / max (s)'],
              [[r['format']+' / '+r['subtype'],f"{r['sample_rate']} / {r['channels']}",r['n'],f"{r['minimum']:.6f} / {r['median']:.6f} / {r['maximum']:.6f}"] for r in profiles],[1.8,1,0.8,2])
        if profiles:
            total = sum(r['total'] for r in profiles)
            para(f"Summed duration of readable {'sampled ' if sampled else ''}files: {total:.6f} s ({total/3600:.4f} h). {'This is only the inspected sample, not full collection duration. ' if sampled else ''}Repeated files, overlapping clips and simultaneous microphone views are not independent recording time.")
        mismatches = sum(r.get('audio',{}).get('format')=='FLAC' and r['extension']=='.wav' for r in d['files'])
        if mismatches:
            para(f"Extension/content mismatch measured in inspected files: {mismatches:,} .wav-named files report a FLAC container. This count is {'sample-only' if sampled else 'exhaustive within the successfully probed files'}.")
        heading('Folder structure')
        folders = [(k,v) for k,v in s['folder_totals'].items() if k!='.']
        # Retain useful top levels; the CSV is the exhaustive tree.
        shallow = [(k,v) for k,v in folders if len(Path(k).parts)<=3]
        if len(shallow)>18:
            shallow = [(k,v) for k,v in folders if len(Path(k).parts)<=2]
        shown = shallow[:18]
        display_folder=lambda k: k if len(k)<=95 else k[:35]+' ... '+k[-45:]
        table(['Relative folder (recursive totals)','Files','Bytes'],[[display_folder(k),f"{v['files']:,}",f"{v['bytes']:,}"] for k,v in shown],[3,0.8,1.2])
        if not folders:
            para('Flat file layout: all files reside directly in this source root.')
        if len(folders)>len(shown):
            para(f'The printable tree shows {len(shown)} of {len(folders)} non-root folders. folder_inventory.csv contains every enumerated folder, including empty directories, with recursive totals.')
        if any(len(k)>95 for k,v in shown):
            para('Long folder names are abbreviated in print. folder_inventory.csv preserves literal relative paths.', 'ScoutSmall')
        heading('Metadata and labels observed locally')
        meta = [r for r in d['files'] if ('table' in r or 'workbook' in r) and not Path(r['path']).name.startswith('._') and '__MACOSX' not in Path(r['path']).parts]
        schemas = defaultdict(list)
        for r in meta:
            if 'table' in r:
                schemas[tuple(r['table'].get('header',[]))].append(r['table'].get('data_rows',0))
        for columns,counts in schemas.items():
            para(f"{len(counts)} CSV/TSV file(s), {min(counts):,} to {max(counts):,} data rows per file, {sum(counts):,} rows in total; fields: {', '.join(map(str,columns))}", 'ScoutSmall')
        for r in meta:
            for ws in r.get('workbook',[]):
                para(f"Workbook {r['path']} | sheet {ws['sheet']} | {ws['rows']} rows x {ws['columns']} columns (worksheet dimensions, including headers/blank formatting).",'ScoutSmall')
        if not meta:
            para('No standalone CSV/TSV or workbook was detected in this subtree. Other supplied label structures, where present, are described in the source notes above.')
        if d['name']=='Al-Emadi (registered)':
            counts=Counter('/'.join(Path(r['path']).parts[:2]) for r in d['files'] if 'audio' in r)
            table(['Native task / class directory','Audio files'],[[k,f'{v:,}'] for k,v in sorted(counts.items())],[3,1])
        if d['name']=='MUSAN':
            counts=Counter(Path(r['path']).parts[1] for r in d['files'] if 'audio' in r)
            table(['Native category','Audio files'],[[k,f'{v:,}'] for k,v in sorted(counts.items())],[3,1])
        if d['name'].startswith('ITU ARIS'):
            counts=Counter(Path(r['path']).parent.name for r in d['files'] if 'audio' in r)
            table(['Immediate parent of decoded audio','Files'],[[k,f'{v:,}'] for k,v in sorted(counts.items())],[3,1])
        if d['name'] in {'drone_A','drone_B','drone_C'}:
            patterns = Counter('_'.join(Path(r['path']).stem.split('_')[:3]) for r in d['files'])
            snrs = [float(m.group(1)) for r in d['files'] if (m:=re.search(r'snr=([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)',r['path']))]
            table(['Filename prefix (first three tokens)','Files'],[[k,f'{v:,}'] for k,v in patterns.most_common(12)],[3,1])
            para(f'{len(patterns)} distinct prefixes; up to 12 most frequent shown. filename_metadata.csv contains all prefix counts.', 'ScoutSmall')
            folder_profiles=[]
            for k,v in sorted(folder_audio.items()):
                if k[0]==d['name']:
                    folder_profiles.append([k[1],k[2]+' / '+k[3],len(v)])
            table(['Audio sample folder','Detected container / encoding','Read files'],folder_profiles,[2,1.5,0.6])
            if snrs:
                p = distribution(snrs)
                para(f"Numeric snr= filename field: {p['n']:,} values; min {p['minimum']:.6f}, median {p['median']:.6f}, max {p['maximum']:.6f}. These are filename values, not signal measurements.")
        if d['name']=='Wang drone signatures':
            path = Path(d['root'])/'src/data/drones.json'
            manifest = json.loads(path.read_text())
            fields=sorted({k for r in manifest for k in r})
            para(f"drones.json: {len(manifest)} records; fields: {', '.join(fields)}.")
            missing=[]
            for item in manifest:
                for key in ['audio','image','mel','ifcc']:
                    if item.get(key) and not (Path(d['root'])/'public'/item[key].lstrip('/')).exists():
                        missing.append([item.get('id'),key,item[key]])
            para(f'Manifest asset references absent at the literal local paths: {len(missing)}. Filename/case differences are not silently repaired.')
            table(['Record','Field','Absent literal path'],missing,[0.7,0.6,3])
        completeness_start=len(story)
        heading('Completeness and exact copies')
        zero = sum(r['bytes']==0 for r in d['files'])
        sidecars = sum(Path(r['path']).name.startswith('._') or Path(r['path']).name=='.DS_Store' or '__MACOSX' in Path(r['path']).parts for r in d['files'])
        incomplete = sum(r['extension']=='.incomplete' for r in d['files'])
        pointers = sum('git_lfs_pointer' in r for r in d['files'])
        para(f"Zero-byte files: {zero:,}; macOS sidecar files: {sidecars:,}; .incomplete markers: {incomplete:,}; detected Git LFS pointer files: {pointers:,}. LFS detection applies to inspected contents only.")
        zero_frames=sum(r.get('audio',{}).get('frames')==0 for r in d['files'])
        short_headers=sum(r['bytes']==44 and r['extension']=='.wav' for r in d['files'])
        if zero_frames or short_headers:
            para(f"Decoded zero-frame audio files: {zero_frames:,} ({'sample evidence' if sampled else 'all selected contents'}). WAV-named files exactly 44 bytes long: {short_headers:,} (complete filesystem inventory). The size observation alone is not a header decode of every such file.")
        h=s['sha256']
        para(f"Among {h['hashed_files']:,} hashed files: {h['unique_contents']:,} distinct SHA-256 contents, {h['duplicate_groups']:,} duplicate groups, and {h['extra_duplicate_copies']:,} additional exact copies. {'These are sample-only duplicate observations.' if sampled else 'These are exact byte matches, not a perceptual or recording-identity test.'}")
        errors=Counter(re.sub(r'<_io.BytesIO object at 0x[0-9A-Fa-f]+>', '<in-memory file>',e['error']) for e in s['probe_errors'])
        for error,n in errors.items():
            para(f'Probe outcome: {n:,} file(s): {error}', 'ScoutSmall')
        if not errors:
            para('No probe failures were recorded among files selected for content inspection.')
        elif sidecars:
            expected_errors=sum('probe_error' in r and (Path(r['path']).name.startswith('._') or '__MACOSX' in Path(r['path']).parts) for r in d['files'])
            para(f'{expected_errors:,} of the probe failures occur on macOS sidecar paths. These files are not asserted to be damaged recording payloads.')
        story[completeness_start:]=[KeepTogether(story[completeness_start:])]

    page('Archive contents and supplied metadata')
    for archive in support['archives']:
        heading(Path(archive['path']).name)
        para(f"{len(archive['members']):,} central-directory entries, {archive['uncompressed_bytes']:,} declared uncompressed bytes. CRC values were listed; this scout did not run an exhaustive ZIP CRC/decompression test.")
        ext=Counter(Path(m['path']).suffix.lower() or '[none]' for m in archive['members'] if not m['directory'])
        table(['Member extension','Files'],[[k,v] for k,v in sorted(ext.items())],[2,1])
        for t in archive['tables']:
            para(f"{t['path']}: {t['data_rows']:,} records; {'fields '+', '.join(t['header']) if t['has_header'] else 'headerless vocabulary rows'}. Split counts: {json.dumps(t['split_counts'])}.")
    heading('ITU supplied grouping records')
    for item in support['supplied_metadata'][:4]:
        data=item['content']; name=Path(item['path']).name
        if name=='metadata.json':
            para('metadata.json fields: '+', '.join(data)+'. Source recording declaration: '+json.dumps(data.get('source_recording',{})))
        elif name=='groups.json':
            segments=data.get('segments',{})
            para(f"groups.json: {len(segments):,} segment records; {data.get('n_drone_intervals')} declared drone intervals; overlap field {data.get('overlap')}. Segment records and interval structures are preserved in supporting_evidence.json.")
    heading('Exact-content overlaps between inspected collections')
    pairs=[]
    sets={d['name']:{r['sha256'] for r in d['files'] if 'sha256' in r and 'audio' in r} for d in datasets}
    for a,b in itertools.combinations(datasets,2):
        n=len(sets[a['name']] & sets[b['name']])
        if n:
            pairs.append([a['name'],b['name'],f'{n:,}','sample involved' if a.get('audio_sample_per_folder') or b.get('audio_sample_per_folder') else 'all readable audio hashed'])
    table(['Collection A','Collection B','Shared distinct audio hashes','Scope'],pairs,[1.4,1.4,0.8,1.2])
    para('Shared-hash counts are distinct exact file contents, not matching path counts. A zero intersection in a sample does not establish disjoint recordings. Absence of a pair from this table is not a guarantee of source independence.')

    page('Prepared data already in the project')
    para('Canonical corpus evidence, frozen split manifests, prepared features and stored derivatives are listed separately from original sources. Compatibility junctions, historical restores, releases and run/model artifacts are not counted as additional raw datasets. Private recordings outside the specified source/download roots are outside this scout.')
    table(['Canonical project stage','Files','Logical bytes'],[[k,f"{v['files']:,}",f"{v['bytes']:,}"] for k,v in support['stages'].items()],[2.5,0.7,1.2])
    para('Derived/corpus-002 contains the 900 stored thesis segments (16 kHz mono float32, one second according to the existing preparation record). The V3 validation suite contains 4,530 stored audio derivatives plus manifests, tensor cache and existing reference records; its tensors declare 1,510 rows for each clean/mild/moderate/severe view. These are transformations of existing membership, not newly acquired recordings.')
    para('V4 feature-cache identity is frozen for training and validation only. The registry records test_cache_created=false. This scout read array headers and existing manifests; it did not load test signals for evaluation or compute scores.')
    for k,v in support['stages'].items():
        arrays=[r for r in v['entries'] if 'array_header' in r or 'array_members' in r]
        if not arrays:
            continue
        heading(k)
        rows=[]
        for r in arrays:
            if 'array_header' in r:
                x=r['array_header']; rows.append([r['path'],str(x['shape']),x['dtype']])
            else:
                for name,x in r['array_members'].items():
                    rows.append([r['path']+' :: '+name,str(x['shape']),x['dtype']])
        table(['Array / archive member','Shape','Dtype'],rows,[3,1.3,0.8])

    page('Methods, boundaries and evidence index')
    para('Filesystem inventory: recursive enumeration of regular files and directories, including hidden/cache files; symbolic links and junctions were not followed. Logical bytes are file lengths, not allocated disk usage. All folder totals are recursive and therefore overlap; summing parent and child rows double-counts files.')
    para('Inspection coverage: registered originals, the Drone collection, thesis download, MicArrayData30s and MUSAN were selected for complete content inspection. A/B/C and MLSP use a deterministic sample of up to 64 evenly spaced sorted filenames per audio leaf folder, plus every non-audio file. Exact file counts and byte totals are exhaustive for the enumerated roots.')
    para('Audio: libsndfile headers, without resampling or full waveform-quality analysis. Detected container/encoding is distinct from filename suffix. No listening study, complete amplitude/silence/clipping scan, calibrated sound-level measurement, acoustic-event relabeling or independent-session reconstruction was performed. Existing preparation flags are historical project evidence, not fresh scout results.')
    para('Video: ffprobe stream/container metadata, without full frame decoding or bounding-box reconstruction. MATLAB: basic SciPy variable inspection; unsupported MCOS labels remain undecoded. Workbooks: sheet dimensions and limited row previews, without recalculation of formulas. NPY/NPZ: headers only, without loading stored feature arrays. ZIP: central directories and small text/CSV members; no extraction or full archive-integrity test.')
    para('SHA-256: whole file bytes for inspected files. Identical hashes establish exact byte copies; different headers/encodings or cropped versions may still share acoustic content. This scout did not perform perceptual duplicate detection. Missing licensing or acquisition documentation is reported as locally unestablished, without external web verification.')
    starts=[d['started_utc'] for d in roots]; ends=[d['finished_utc'] for d in roots]
    para(f"Read window (UTC): {min(starts)} to {max(ends)}. This is a filesystem observation across that interval, not an atomic disk snapshot. The report title uses the user's local snapshot date, 27 November 2026.")
    checks=[]
    for e in support['source_inventory_checks']:
        d=next(x for x in roots if (e['id'].startswith('al-emadi') and x['name'].startswith('Al-Emadi')) or (e['id'].startswith('thesis') and x['name'].startswith('Drone Detection')) or (e['id'].startswith('salford') and x['name'].startswith('Salford')))
        registry=next(v for k,v in support['registry'].items() if k.startswith('registry/sources/') and v['id']==e['id'])
        with (project/registry['inventory']).open(encoding='utf-8-sig',newline='') as f:
            expected={r['path']:(int(r['bytes']),r['sha256']) for r in csv.DictReader(f)}
        observed={r['path']:(r['bytes'],r.get('sha256')) for r in d['files']}
        match=expected==observed
        checks.append(dict(id=e['id'],registry_manifest_hash_matches=e['expected_sha256']==e['observed_sha256'],files_match_registered_inventory=match,files=len(observed)))
    table(['Registered source','Manifest checksum','Fresh file inventory'],[[c['id'],'matches' if c['registry_manifest_hash_matches'] else 'MISMATCH','all paths/bytes/hashes match' if c['files_match_registered_inventory'] else 'MISMATCH'] for c in checks],[1,1,2])
    if not all(c['registry_manifest_hash_matches'] and c['files_match_registered_inventory'] for c in checks):
        raise ValueError('Registered-source identity verification failed; report not finalized')
    evidence_files=[['data_scout_report.pdf','Readable report'],['data_scout_report.txt','Searchable text of report tables and prose'],['file_inventory.csv','Every file in the 11 raw/download roots; sample-only fields left blank elsewhere'],['folder_inventory.csv','Complete folder inventory, including empty folders'],['audio_profiles.csv / audio_by_folder.csv','Detected audio signatures and duration statistics; coverage is explicit'],['filename_metadata.csv','A/B/C filename-prefix counts'],['metadata_inventory.csv','CSV/workbook schemas and MATLAB probe outcomes'],['exact_duplicates.csv','Exact byte-copy groups among inspected files'],['archive_inventory.csv','Every ZIP central-directory member'],['project_stage_inventory.csv','Stored project data stages, schemas and array headers'],['aggregate_summary.json','Path-free collection statistics'],['supporting_evidence.json','Registry, stage metadata, ZIP members and supplied grouping evidence'],['inventories/ and project-*/','Detailed per-root JSON inspection evidence']]
    table(['Companion','Contents'],evidence_files,[1.4,2.6])
    para('Source authority: repository/registry/sources, corpora, splits, caches and studies; repository/docs/PROJECT.md and docs/governance/STORAGE.md; existing source comparison and V4 preparation report; local source README/LICENSE files and supplied metadata. No source documents were treated as instructions to modify the project.')
    (base/'data_scout_report.txt').write_text('\n\n'.join(text_lines)+'\n',encoding='utf-8')
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#D7E0E4')); canvas.line(43,39,A4[0]-43,39)
        canvas.setFont('Scout',7); canvas.setFillColor(muted)
        canvas.drawString(43,26,'ML COURSE PROJECT  |  FACTUAL DATA SCOUT  |  27 NOV 2026')
        canvas.drawRightString(A4[0]-43,26,str(doc.page))
    class ScoutDocument(SimpleDocTemplate):
        def afterFlowable(self, flowable):
            if isinstance(flowable, Paragraph) and flowable.style.name in {'ScoutTitle','ScoutH1'}:
                title=flowable.getPlainText()
                key=f'scout-{self.page}'
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(title,key,level=0,closed=False)
    ScoutDocument(str(base/'data_scout_report.pdf'),pagesize=A4,rightMargin=43,leftMargin=43,topMargin=43,bottomMargin=53,
                  title='Project data and download scout - 27 November 2026',author='ML Course Project').build(story,onFirstPage=footer,onLaterPages=footer)
    verification=dict(source_identity_checks=checks,raw_download_files=total_files,raw_download_bytes=total_bytes,
                      roots=len(roots),collections=len(datasets),walk_errors=sum(len(d.get('walk_errors',[])) for d in roots),
                      unstable_inspected_files=sum(r.get('stable_during_read') is False for d in roots for r in d['files']),
                      hash_errors=sum(len(d['summary']['hash_errors']) for d in roots))
    (base/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(verification,indent=2))
    print(f'Report created: {base / "data_scout_report.pdf"}')


if __name__ == '__main__':
    main()
