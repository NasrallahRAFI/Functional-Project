"""PCM WAV reading and deterministic conversion to mono 16 kHz PCM WAV."""

from __future__ import annotations

import wave
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .resample import resample, resample_stream


@dataclass(frozen=True)
class WavAudio:
    samples: np.ndarray
    sample_rate: int
    channels: int
    sample_width: int
    clipped_fraction: float

    @property
    def clip_fraction(self) -> float:
        return self.clipped_fraction


@dataclass(frozen=True)
class WavHeader:
    format_tag: int
    channels: int
    sample_rate: int
    bits_per_sample: int
    block_align: int
    data_offset: int
    data_bytes: int

    @property
    def frames(self) -> int:
        return self.data_bytes // self.block_align


@dataclass(frozen=True)
class StandardizedWavInfo:
    sample_rate: int
    channels: int
    sample_width: int
    clipped_fraction: float

    @property
    def clip_fraction(self) -> float:
        return self.clipped_fraction


def inspect_wav_header(path: str | Path) -> WavHeader:
    """Read RIFF fmt/data fields without loading audio payload bytes."""
    with Path(path).open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError("expected a little-endian RIFF/WAVE file")
        fmt = None
        data_offset = data_bytes = None
        while True:
            chunk = stream.read(8)
            if len(chunk) != 8:
                break
            chunk_id, size = chunk[:4], int.from_bytes(chunk[4:], "little")
            offset = stream.tell()
            if chunk_id == b"fmt ":
                payload = stream.read(min(size, 64))
                if len(payload) < 16:
                    raise ValueError("truncated WAV fmt chunk")
                tag, channels, rate, _byte_rate, align, bits = struct.unpack_from("<HHIIHH", payload)
                if tag == 0xFFFE and len(payload) >= 40:
                    guid = payload[24:40]
                    if guid == bytes.fromhex("0100000000001000800000aa00389b71"):
                        tag = 1
                    elif guid == bytes.fromhex("0300000000001000800000aa00389b71"):
                        tag = 3
                fmt = (tag, channels, rate, bits, align)
            elif chunk_id == b"data":
                data_offset, data_bytes = offset, size
            stream.seek(offset + size + (size & 1))
            if fmt is not None and data_offset is not None:
                break
    if fmt is None or data_offset is None or data_bytes is None:
        raise ValueError("WAV is missing a fmt or data chunk")
    tag, channels, rate, bits, align = fmt
    if channels <= 0 or rate <= 0 or align <= 0 or data_bytes % align:
        raise ValueError("invalid WAV channel/rate/alignment metadata")
    return WavHeader(tag, channels, rate, bits, align, data_offset, data_bytes)


def _decode_channel(raw: bytes, header: WavHeader, count: int, channel_index: int) -> np.ndarray:
    width = (header.bits_per_sample + 7) // 8
    if width * header.channels != header.block_align:
        raise ValueError("unsupported WAV block alignment")
    if header.format_tag == 1:
        decoded = _decode_pcm(raw, width, count, header.channels)
    elif header.format_tag == 3 and header.bits_per_sample == 32:
        decoded = np.frombuffer(raw, dtype="<f4").reshape(count, header.channels)
    elif header.format_tag == 3 and header.bits_per_sample == 64:
        decoded = np.frombuffer(raw, dtype="<f8").reshape(count, header.channels).astype(np.float32)
    else:
        raise ValueError(f"unsupported WAV format tag={header.format_tag}, bits={header.bits_per_sample}")
    values = decoded[:, channel_index]
    if not np.all(np.isfinite(values)):
        raise ValueError("WAV contains non-finite samples")
    return values.astype(np.float32, copy=False)


def _read_channel_frames(path: str | Path, header: WavHeader, start: int, count: int,
                         channel_index: int) -> np.ndarray:
    if not 0 <= channel_index < header.channels or start < 0 or count < 0 or start + count > header.frames:
        raise ValueError("requested WAV channel/frame range is invalid")
    with Path(path).open("rb") as stream:
        stream.seek(header.data_offset + start * header.block_align)
        raw = stream.read(count * header.block_align)
    if len(raw) != count * header.block_align:
        raise ValueError("WAV data ended before the declared frame range")
    return _decode_channel(raw, header, count, channel_index)


def standardize_wav_streaming(
    source: str | Path,
    destination: str | Path,
    *,
    target_rate: int = 16_000,
    channel_index: int = 0,
    segment_start_seconds: float = 0.0,
    segment_end_seconds: float | None = None,
    max_clip_fraction: float | None = None,
    output_block_frames: int = 8192,
) -> StandardizedWavInfo:
    """Select one channel and resample a segment while reading bounded WAV blocks."""
    header = inspect_wav_header(source)
    if header.format_tag not in {1, 3}:
        raise ValueError(f"unsupported WAV format tag {header.format_tag}")
    start = int(round(segment_start_seconds * header.sample_rate))
    stop = header.frames if segment_end_seconds is None else int(round(segment_end_seconds * header.sample_rate))
    if start < 0 or stop <= start or stop > header.frames or target_rate <= 0:
        raise ValueError("segment or target sample rate is outside source WAV")
    if not 0 <= channel_index < header.channels or output_block_frames <= 0:
        raise ValueError("channel index or output block size is invalid")
    count = stop - start
    clipped, scanned = 0, 0
    for offset in range(start, stop, 65_536):
        block = _read_channel_frames(source, header, offset, min(65_536, stop - offset), channel_index)
        clipped += int(np.count_nonzero(np.abs(block) >= 0.999))
        scanned += block.size
    fraction = clipped / max(scanned, 1)
    if max_clip_fraction is not None and not 0 <= max_clip_fraction <= 1:
        raise ValueError("max_clip_fraction must be between 0 and 1, or None")
    if max_clip_fraction is not None and fraction > max_clip_fraction:
        raise ValueError(f"WAV is clipped ({fraction:.3%} near-full-scale frames)")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite WAV: {destination}")
    try:
        converter = None if target_rate == header.sample_rate else resample_stream(header.sample_rate, target_rate, quality="HQ")
        with wave.open(str(destination), "wb") as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(target_rate)
            for offset in range(start, stop, output_block_frames):
                source_block = _read_channel_frames(source, header, offset, min(output_block_frames, stop - offset), channel_index)
                converted = source_block if converter is None else converter.resample_chunk(source_block, last=False)
                pcm = np.rint(np.clip(converted, -1.0, 1.0) * 32767.0).astype("<i2")
                stream.writeframesraw(pcm.tobytes())
            if converter is not None:
                tail = converter.resample_chunk(np.empty(0, dtype=np.float32), last=True)
                pcm = np.rint(np.clip(tail, -1.0, 1.0) * 32767.0).astype("<i2")
                stream.writeframesraw(pcm.tobytes())
        return StandardizedWavInfo(target_rate, 1, 2, float(fraction))
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _decode_pcm(raw: bytes, sample_width: int, frame_count: int, channels: int) -> np.ndarray:
    if sample_width == 1:
        values = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        values = (values - 128.0) / 128.0
    elif sample_width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        integers = packed[:, 0].astype(np.int32) | (packed[:, 1].astype(np.int32) << 8) | (packed[:, 2].astype(np.int32) << 16)
        integers = (integers ^ 0x800000) - 0x800000
        values = integers.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float64).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported PCM sample width: {sample_width} bytes")
    if values.size != frame_count * channels:
        raise ValueError("WAV payload length does not match its frame header")
    return values.reshape(frame_count, channels)


def _is_integer_pcm_wav(path: str | Path) -> bool:
    """Inspect RIFF fmt metadata so IEEE-float WAV is not silently decoded as integer PCM."""
    with Path(path).open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            return False
        while True:
            chunk = stream.read(8)
            if len(chunk) < 8:
                return False
            chunk_id, chunk_size = chunk[:4], int.from_bytes(chunk[4:], "little")
            if chunk_id == b"fmt ":
                fmt = stream.read(chunk_size)
                if len(fmt) < 16:
                    return False
                tag = int.from_bytes(fmt[:2], "little")
                if tag == 1:
                    return True
                if tag == 0xFFFE and len(fmt) >= 40:
                    pcm_guid = bytes.fromhex("0100000000001000800000aa00389b71")
                    return fmt[24:40] == pcm_guid
                return False
            stream.seek(chunk_size + (chunk_size & 1), 1)


def read_wav_mono(path: str | Path, channel_index: int | None = 0, *,
                  segment_start_seconds: float | None = None,
                  segment_end_seconds: float | None = None) -> WavAudio:
    """Read uncompressed PCM WAV, selecting one channel or averaging channels, and flag clipping."""
    info = sf.info(str(path))
    channels, sample_rate, frames = info.channels, info.samplerate, info.frames
    if channels <= 0 or sample_rate <= 0:
        raise ValueError("invalid audio header")
    if any(value is not None and not np.isfinite(value)
           for value in (segment_start_seconds, segment_end_seconds)):
        raise ValueError("segment time bounds must be finite")
    decoded, decoded_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if decoded_rate != sample_rate or decoded.shape != (frames, channels):
        raise ValueError("audio data does not match its header")
    start = 0 if segment_start_seconds is None else int(round(segment_start_seconds * sample_rate))
    stop = frames if segment_end_seconds is None else int(round(segment_end_seconds * sample_rate))
    if start < 0 or stop <= start or stop > frames:
        raise ValueError("segment time bounds are invalid or outside the source WAV")
    if channel_index is None:
        mono = decoded.mean(axis=1, dtype=np.float64).astype(np.float32)
        clipping_signal = decoded[start:stop]
    else:
        if not 0 <= channel_index < channels:
            raise ValueError(f"channel_index {channel_index} outside WAV's {channels} channels")
        mono = decoded[:, channel_index].astype(np.float32)
        clipping_signal = decoded[start:stop, channel_index:channel_index + 1]
    clipped_fraction = float(np.mean(np.max(np.abs(clipping_signal), axis=1) >= 0.999)) if frames else 0.0
    return WavAudio(mono, sample_rate, channels, 4, clipped_fraction)


def write_wav_pcm16(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1 or sample_rate <= 0 or not np.all(np.isfinite(x)):
        raise ValueError("audio must be a finite mono vector and sample_rate positive")
    clipped = np.clip(x, -1.0, 1.0)
    pcm = np.rint(clipped * 32767.0).astype("<i2")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite WAV: {output}")
    with wave.open(str(output), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


def standardize_wav(
    source: str | Path,
    destination: str | Path,
    *,
    target_rate: int = 16_000,
    max_clip_fraction: float | None = None,
    channel_index: int | None = None,
    segment_start_seconds: float | None = None,
    segment_end_seconds: float | None = None,
) -> WavAudio:
    """Convert to mono 16-bit PCM at target rate; reject corrupt or materially clipped input."""
    if max_clip_fraction is not None and not 0 <= max_clip_fraction <= 1:
        raise ValueError("max_clip_fraction must be between 0 and 1, or None")
    audio = read_wav_mono(source, channel_index=channel_index,
                           segment_start_seconds=segment_start_seconds,
                           segment_end_seconds=segment_end_seconds)
    if audio.samples.size == 0:
        raise ValueError("WAV contains no audio frames")
    if any(value is not None and not np.isfinite(value) for value in (segment_start_seconds, segment_end_seconds)):
        raise ValueError("segment time bounds must be finite")
    start = 0 if segment_start_seconds is None else int(round(segment_start_seconds * audio.sample_rate))
    stop = audio.samples.size if segment_end_seconds is None else int(round(segment_end_seconds * audio.sample_rate))
    if start < 0 or stop <= start or stop > audio.samples.size:
        raise ValueError("segment time bounds are invalid or outside the source WAV")
    selected = audio.samples[start:stop]
    if selected.size == 0:
        raise ValueError("selected WAV segment contains no samples")
    clip_fraction = audio.clipped_fraction
    if max_clip_fraction is not None and clip_fraction > max_clip_fraction:
        raise ValueError(f"WAV is clipped ({clip_fraction:.3%} near-full-scale frames)")
    converted = resample(selected, audio.sample_rate, target_rate)
    write_wav_pcm16(destination, converted, target_rate)
    return WavAudio(converted, target_rate, 1, 2, clip_fraction)
