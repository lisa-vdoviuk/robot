"""Streaming Silero ONNX adapter for the existing HEAR recorder."""
import hashlib
import threading
import time
from pathlib import Path
import numpy as np

MODEL = Path(__file__).resolve().parent / 'models' / 'silero_vad.onnx'
SHA256 = '1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3'
COMMIT = '1e261b036686cd0017d500ee96acd1c4ba572a9d'
IMPORT_ERROR = ''
try:
    import onnxruntime as ort
    import scipy
    from scipy.signal import firwin, lfilter
except ImportError:
    IMPORT_ERROR = 'Install numpy scipy onnxruntime in the server Python environment'
if not MODEL.is_file():
    IMPORT_ERROR += ' Missing backend/models/silero_vad.onnx; install model and restart server.'

CONFIG = dict(engine='Silero ONNX', source_commit=COMMIT, model_sha256=SHA256,
              threshold=0.5, input_rate=48000, model_rate=16000, frame_ms=32,
              start_positive_frames=3, end_negative_frames=10,
              resampling='Causal FIR 127 taps, Hamming, cutoff 7200 Hz; decimation by 3; persistent state',
              filter_delay_ms=1.3125,
              timing='Filtered sample time; no compensation for filter, device or browser latency',
              tail='Final incomplete 32 ms frame is not classified; raw WAV retained',
              playback='Approximate server playback hints; no echo cancellation')
_session = None
_lock = threading.Lock()


def get_session():
    global _session
    if IMPORT_ERROR:
        raise RuntimeError(IMPORT_ERROR)
    with _lock:
        if _session is None:
            if hashlib.sha256(MODEL.read_bytes()).hexdigest() != SHA256:
                raise RuntimeError('Silero model checksum mismatch')
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            _session = ort.InferenceSession(str(MODEL), sess_options=options,
                                           providers=['CPUExecutionProvider'])
    return _session


class Detector:
    def __init__(self):
        self.session = get_session()
        self.taps = firwin(127, 7200, fs=48000, window='hamming')
        self.filter_state = np.zeros(126)
        self.input_samples = 0
        self.buffer = np.empty(0, dtype=np.float32)
        self.flags = np.empty(0, dtype=bool)
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)
        self.index = self.positive = self.negative = self.raw_positive = 0
        self.event = None
        self.events = []
        self.times = []
        self.pending_playback = False
        self.last_end = 0.0

    def feed(self, raw, playback=False):
        if len(raw) % 2:
            raise ValueError('Incomplete PCM16 sample')
        if not raw:
            return []
        x = np.frombuffer(raw, '<i2').astype(np.float64) / 32768
        filtered, self.filter_state = lfilter(self.taps, [1.0], x, zi=self.filter_state)
        # Preserve decimation phase across arbitrary capture block boundaries.
        y = filtered[(-self.input_samples) % 3::3].astype(np.float32)
        self.input_samples += len(x)
        self.buffer = np.concatenate((self.buffer, y))
        self.flags = np.concatenate((self.flags, np.full(len(y), playback, dtype=bool)))
        rows = []
        while len(self.buffer) >= 512:
            chunk, self.buffer = self.buffer[:512], self.buffer[512:]
            own = bool(np.any(self.flags[:512]))
            self.flags = self.flags[512:]
            model_input = np.concatenate((self.context, chunk[None, :]), axis=1)
            before = time.perf_counter()
            score, self.state = self.session.run(None, {
                'input': model_input, 'state': self.state, 'sr': np.array(16000, dtype=np.int64)})
            elapsed = (time.perf_counter() - before) * 1000
            self.context = model_input[:, -64:].copy()
            score = float(score.item())
            if not np.isfinite(score):
                raise RuntimeError('Silero returned a non-finite score')
            self.times.append(elapsed)
            speech = score >= CONFIG['threshold']
            self.raw_positive += int(speech)
            start, end = self.index * .032, (self.index + 1) * .032
            if speech:
                if self.positive == 0:
                    self.pending_playback = False
                self.positive += 1
                self.negative = 0
                self.pending_playback |= own
                self.last_end = end
                if self.event is None and self.positive >= 3:
                    self.event = dict(start_s=round(start-.064, 6),
                                      detected_at_s=round(end, 6),
                                      playback_possible=self.pending_playback)
            else:
                self.positive = 0
                self.pending_playback = False
                self.negative += 1
            if self.event is not None:
                self.event['playback_possible'] |= own
                if self.negative >= 10:
                    self.close_event(end, 'silence')
            rows.append(dict(start_s=round(start, 6), end_s=round(end, 6),
                             speech_score=score, speech_raw=speech,
                             speech_active=self.event is not None,
                             playback_possible=own, process_ms=elapsed))
            self.index += 1
        return rows

    def close_event(self, at, reason):
        if self.event is not None:
            self.event.update(end_s=round(self.last_end, 6),
                              closed_at_s=round(at, 6), close_reason=reason)
            self.events.append(self.event)
            self.event = None

    def finish(self):
        self.close_event(self.index*.032, 'recording_end')
        return dict(config={**CONFIG, 'onnxruntime': ort.__version__, 'scipy': scipy.__version__},
                    frames=self.index, raw_positive_frames=self.raw_positive,
                    raw_positive_fraction=self.raw_positive/max(self.index, 1),
                    unprocessed_tail_samples=len(self.buffer), tail_sample_rate=16000,
                    input_samples=self.input_samples, events=self.events,
                    processing_ms_median=float(np.median(self.times)) if self.times else None,
                    processing_ms_p95=float(np.percentile(self.times, 95)) if self.times else None,
                    interpretation='Model scores and predicted intervals, not accuracy or confirmed human presence. Timings cover ONNX calls only. Raw 48 kHz WAV is unchanged.')
