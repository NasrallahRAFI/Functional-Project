"""Front-end streaming demo with a deliberately stubbed detector."""
from __future__ import annotations
import argparse
import logging
import queue
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dronedet.config import load_config
from dronedet.features import clip_flags, load_audio, mel_power, to_db, transform
from dronedet.resample import resample, resample_stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("dronedet.demo")


class StubScorer:
    def score(self, feature: np.ndarray) -> float:
        return float(np.clip(np.mean(np.abs(feature)), 0, 1))


def _stats_for_stub(config, stats):
    # A neutral transform keeps the non-model demo executable before train stats exist.
    return stats if stats is not None else {"mean": np.zeros(config.n_mels, np.float32),
                                            "std": np.ones(config.n_mels, np.float32)}


class VoteSmoother:
    def __init__(self, count: int): self.count=count; self.values=[]
    def push(self, score: float) -> float:
        self.values=(self.values+[float(score)])[-self.count:]
        return float(np.mean(self.values))


class Ring:
    def __init__(self, size: int): self.size=size; self.data=np.empty(0,np.float32)
    def push(self, x):
        self.data=np.concatenate((self.data,np.asarray(x,np.float32)))
        if len(self.data)>self.size: self.data=self.data[-self.size:]
        return len(self.data)==self.size
    def reset(self): self.data=np.empty(0,np.float32)


class LiveWindowBuilder:
    """Streaming resampler/ring that suppresses output after a discontinuity."""
    def __init__(self, config, source_rate: int):
        self.config=config; self.source_rate=source_rate; self.reset()
    def reset(self):
        self.converter=None if self.source_rate==self.config.sr else resample_stream(
            self.source_rate,self.config.sr,quality=self.config.resample_quality)
        self.ring=Ring(self.config.win_samples); self.pending=np.empty(0,np.float32); self.samples_since_reset=0
        self.next_window_end=self.config.win_samples; self.emitted_starts=[]
    def push(self, block: np.ndarray, *, overflow: bool=False) -> list[np.ndarray]:
        if overflow:
            self.reset(); return []
        x=np.asarray(block,np.float32)
        if x.ndim!=1 or not np.all(np.isfinite(x)):
            self.reset(); return []
        converted=x if self.converter is None else self.converter.resample_chunk(x,last=False)
        return self._consume(converted)

    def finish(self) -> list[np.ndarray]:
        """Flush the streaming resampler and emit any final complete windows."""
        if self.converter is None:
            return self._consume(np.empty(0,np.float32))
        tail=self.converter.resample_chunk(np.empty(0,np.float32),last=True)
        self.converter=None
        return self._consume(tail)

    def _consume(self, converted: np.ndarray) -> list[np.ndarray]:
        self.pending=np.concatenate((self.pending,np.asarray(converted,np.float32)))
        windows=[]; hop=self.config.hop
        cadence=max(hop,int(round(self.config.decision_interval_s*self.config.sr)))
        while len(self.pending)>=hop:
            piece=self.pending[:hop]; self.pending=self.pending[hop:]
            self.samples_since_reset+=len(piece)
            ready=self.ring.push(piece)
            if ready and self.samples_since_reset>=self.next_window_end:
                windows.append(self.ring.data.copy())
                self.emitted_starts.append(self.samples_since_reset-self.config.win_samples)
                self.next_window_end+=cadence
        return windows


def check_stream_parity(config, *, seed: int = 20261123, duration_s: float = 6.0) -> dict:
    """Deterministic G14 T1-T3 streaming parity and discontinuity contract."""
    rng=np.random.default_rng(seed)
    n_source=int(round(duration_s*config.sr))
    clean=rng.normal(0.0,0.08,n_source).astype(np.float32)
    builder=LiveWindowBuilder(config,config.sr)
    t1=[]
    for start in range(0,len(clean),1000): t1.extend(builder.push(clean[start:start+1000]))
    t1.extend(builder.finish())
    expected16=[clean[start:start+config.win_samples] for start in builder.emitted_starts]
    t1_ok=len(t1)==len(expected16) and all(np.array_equal(a,b) for a,b in zip(t1,expected16))

    source48=rng.normal(0.0,0.08,int(round(duration_s*48_000))).astype(np.float32)
    offline=resample(source48,48_000,config.sr,quality=config.resample_quality)
    sizes=(160,1000,4096,12000)
    t2={}
    for chunk_size in sizes:
        stream=resample_stream(48_000,config.sr,quality=config.resample_quality)
        parts=[stream.resample_chunk(source48[start:start+chunk_size],last=False)
               for start in range(0,len(source48),chunk_size)]
        parts.append(stream.resample_chunk(np.empty(0,np.float32),last=True))
        streamed=np.concatenate(parts).astype(np.float32)
        n=min(len(offline),len(streamed))
        max_resample_error=float(np.max(np.abs(offline[:n]-streamed[:n]),initial=0.0))
        length_match=len(offline)==len(streamed)
        # Measure any startup offset over a 4096-sample prefix, if exact alignment fails.
        candidates=[]
        for lag in range(-32,33):
            a0=max(0,lag); b0=max(0,-lag); width=min(4096,len(offline)-a0,len(streamed)-b0)
            candidates.append((float(np.max(np.abs(offline[a0:a0+width]-streamed[b0:b0+width]),initial=0.0)),lag))
        prefix_error,offset=min(candidates)
        live=LiveWindowBuilder(config,48_000); windows=[]
        for start in range(0,len(source48),chunk_size): windows.extend(live.push(source48[start:start+chunk_size]))
        windows.extend(live.finish())
        expected=[offline[start:start+config.win_samples] for start in live.emitted_starts]
        db_errors=[]
        if len(windows)==len(expected):
            for actual_window,expected_window in zip(windows,expected):
                actual_db=to_db(mel_power(actual_window,config),config)
                expected_db=to_db(mel_power(expected_window,config),config)
                db_errors.append(float(np.max(np.abs(actual_db-expected_db),initial=0.0)))
        max_db_error=max(db_errors,default=float("inf"))
        passed=(length_match and max_resample_error<=1e-6 and offset==0 and
                len(windows)==len(expected) and max_db_error<=1e-3)
        t2[str(chunk_size)]={"stream_samples":len(streamed),"offline_samples":len(offline),
            "max_resampled_signal_abs_difference":max_resample_error,
            "startup_offset_samples":offset,"prefix_max_abs_difference":prefix_error,
            "windows_streamed":len(windows),"windows_offline":len(expected),
            "max_logmel_abs_difference_db":max_db_error,"pass":passed}

    recovery=LiveWindowBuilder(config,48_000)
    pre=rng.normal(0.0,0.08,48_000).astype(np.float32)
    for start in range(0,len(pre),4096): recovery.push(pre[start:start+4096])
    old_converter=recovery.converter
    cleared=recovery.push(np.zeros(4096,np.float32),overflow=True)
    ring_cleared=not recovery.ring.data.size
    reset_ok=(not cleared and ring_cleared and recovery.samples_since_reset==0 and
              recovery.converter is not None and recovery.converter is not old_converter)
    post=rng.normal(0.0,0.08,48_000).astype(np.float32)
    post_windows=[]
    for start in range(0,len(post),4096): post_windows.extend(recovery.push(post[start:start+4096]))
    post_windows.extend(recovery.finish())
    post_offline=resample(post,48_000,config.sr,quality=config.resample_quality)
    first_match=bool(post_windows and recovery.emitted_starts and recovery.emitted_starts[0]==0
                     and np.array_equal(post_windows[0],post_offline[:config.win_samples]))
    t3_ok=reset_ok and first_match
    return {"status":"PASS" if t1_ok and all(x["pass"] for x in t2.values()) and t3_ok else "STOP",
            "seed":seed,"t1_16khz":{"windows":len(t1),"expected_windows":len(expected16),"exact":t1_ok},
            "t2_48khz":t2,"t3_overflow_recovery":{"ring_cleared":ring_cleared,
                "resampler_replaced":reset_ok,"first_post_gap_window_matches_offline":first_match,
                "fresh_samples_before_first_window":(recovery.emitted_starts[0]+config.win_samples
                                                       if recovery.emitted_starts else None),
                "pass":t3_ok}}


def file_mode(path: str, config, stats=None, chunk_size=2048):
    audio, source_rate=load_audio(path,config)
    if not np.all(np.isfinite(audio)):
        LOG.warning("file contains non-finite audio; skipping it in demo mode")
        return []
    stats=_stats_for_stub(config,stats)
    builder=LiveWindowBuilder(config,source_rate); scorer=StubScorer(); smoother=VoteSmoother(config.smoothing_windows); outputs=[]
    windows=[]
    for start in range(0,len(audio),chunk_size): windows.extend(builder.push(audio[start:start+chunk_size]))
    windows.extend(builder.finish())
    for i,window in enumerate(windows):
        flags=clip_flags(window,config)
        feat=transform(mel_power(window,config),config,stats)
        end_sample=builder.emitted_starts[i]+config.win_samples
        score=smoother.push(scorer.score(feat)); outputs.append(((end_sample-config.win_samples)/config.sr,score,flags))
        LOG.info("stub score=%.4f flags=%s",score,flags)
    return outputs


def live_mode(config, stats=None):
    import sounddevice as sd
    audio_queue: queue.Queue = queue.Queue(maxsize=32)
    def callback(indata, frames, time_info, status):
        if status:
            LOG.warning("capture status: %s",status)
        try: audio_queue.put_nowait((indata[:,0].copy(),status))
        except queue.Full: LOG.warning("capture queue overflow; dropping block")
    device_rate=int(sd.query_devices(kind="input")["default_samplerate"])
    builder=LiveWindowBuilder(config,device_rate)
    stats=_stats_for_stub(config,stats)
    scorer=StubScorer(); smoother=VoteSmoother(config.smoothing_windows)
    with sd.InputStream(samplerate=device_rate,channels=1,dtype="float32",callback=callback):
        LOG.info("live capture at %d Hz; Ctrl+C to stop",device_rate)
        while True:
            block,status=audio_queue.get()
            # PortAudio's sounddevice overflow status is named input_overflow.
            if status and getattr(status,"input_overflow",False):
                LOG.warning("input_overflow: reset resampler and ring buffer")
                builder.reset(); smoother.values=[]; continue
            windows=builder.push(block)
            if not np.all(np.isfinite(block)):
                LOG.warning("skipping non-finite capture block"); smoother.values=[]; continue
            for window in windows:
                started=time.perf_counter()
                flags=clip_flags(window,config)
                feat=transform(mel_power(window,config),config,stats)
                score=smoother.push(scorer.score(feat))
                elapsed=(time.perf_counter()-started)*1000
                LOG.info("stub score %.4f flags=%s front-end %.2f ms",score,flags,elapsed)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default=str(Path(__file__).resolve().parents[1]/"config.yaml")); parser.add_argument("--file"); parser.add_argument("--live",action="store_true")
    args=parser.parse_args(); cfg=load_config(args.config)
    if args.live: live_mode(cfg)
    elif args.file:
        outputs=file_mode(args.file,cfg); print(f"Processed {len(outputs)} windows with stub detector")
    else: parser.error("choose --file WAV or --live")
if __name__=="__main__": main()
