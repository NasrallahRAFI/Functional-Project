import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SOURCE))

from drone_audio.audio_io import read_wav_mono, standardize_wav, write_wav_pcm16  # noqa: E402


class AudioIoTests(unittest.TestCase):
    def test_wav_write_read_and_standardize(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            target = Path(directory) / "standard.wav"
            audio = (0.2 * np.sin(2 * np.pi * 440 * np.arange(8_000) / 8_000)).astype(np.float32)
            write_wav_pcm16(source, audio, 8_000)
            decoded = read_wav_mono(source)
            self.assertEqual(decoded.sample_rate, 8_000)
            self.assertEqual(decoded.channels, 1)
            prepared = standardize_wav(source, target)
            self.assertEqual(prepared.sample_rate, 16_000)
            self.assertEqual(prepared.samples.size, 16_000)
            self.assertTrue(target.is_file())

            clipped_target = Path(directory) / "middle.wav"
            middle = standardize_wav(source, clipped_target, segment_start_seconds=0.25, segment_end_seconds=0.75)
            self.assertEqual(middle.samples.size, 8_000)

    def test_selecting_a_multichannel_input_uses_only_that_channel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "array.wav"
            frames = 1000
            left = np.full(frames, 32767, dtype=np.int16)
            right = (np.sin(np.arange(frames) * 0.1) * 5000).astype(np.int16)
            interleaved = np.column_stack((left, right)).astype("<i2")
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(2)
                stream.setsampwidth(2)
                stream.setframerate(16_000)
                stream.writeframes(interleaved.tobytes())
            selected = read_wav_mono(path, channel_index=1)
            self.assertEqual(selected.channels, 2)
            self.assertEqual(selected.clipped_fraction, 0.0)
            self.assertGreater(float(np.max(np.abs(selected.samples))), 0.1)


if __name__ == "__main__":
    unittest.main()
