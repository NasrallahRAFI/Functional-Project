"""Checks for the inventory's read-only and sampled-evidence contracts."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'scout_local_data.py'
spec = importlib.util.spec_from_file_location('data_scout', SCRIPT)
scout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scout)


class ScoutTests(unittest.TestCase):
    def test_sampling_preserves_complete_inventory_and_source_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'input'
            root.mkdir()
            for i in range(5):
                with wave.open(str(root / f'{i}.wav'), 'wb') as stream:
                    stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                    stream.writeframes(struct.pack('<160h', *([i] * 160)))
            (root / 'nested').mkdir()
            (root / 'nested' / 'empty.txt').touch()
            (root / 'pointer.wav').write_text('version https://git-lfs.github.com/spec/v1\noid sha256:' + '0' * 64 + '\nsize 123\n')
            before = {p.relative_to(root).as_posix(): (scout.digest(p), p.stat().st_mtime_ns)
                      for p in root.rglob('*') if p.is_file()}
            out = Path(tmp) / 'inventory.json'
            scout.scan(root, out, workers=2, audio_sample_per_folder=3)
            data = json.loads(out.read_text())
            self.assertEqual(data['summary']['files'], 7)
            self.assertEqual(data['summary']['bytes'], sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
            self.assertEqual(data['summary']['audio']['metadata_only_files'], 3)
            self.assertEqual(data['summary']['audio']['readable_files'], 2)
            self.assertEqual(sum('git_lfs_pointer' in r for r in data['files']), 1)
            self.assertEqual({r['audio']['sample_rate'] for r in data['files'] if 'audio' in r}, {16000})
            after = {p.relative_to(root).as_posix(): (scout.digest(p), p.stat().st_mtime_ns)
                     for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(data['walk_errors'], [])
            self.assertEqual(data['summary']['probe_errors'], [])

    def test_full_scan_counts_exact_byte_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'input'
            root.mkdir()
            (root / 'a.txt').write_text('same content')
            (root / 'b.txt').write_text('same content')
            (root / 'c.txt').write_text('different')
            out = Path(tmp) / 'inventory.json'
            scout.scan(root, out, workers=2)
            data = json.loads(out.read_text())
            self.assertEqual(data['summary']['sha256']['unique_contents'], 2)
            self.assertEqual(data['summary']['sha256']['extra_duplicate_copies'], 1)
            self.assertTrue(all(r['stable_during_read'] for r in data['files']))


if __name__ == '__main__':
    unittest.main()
