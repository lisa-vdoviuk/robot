"""WebRTC VAD baseline; timestamps refer to recorded samples, not wall time."""
import time
import numpy as np
try:
    import webrtcvad
    IMPORT_ERROR = ''
except ImportError:
    webrtcvad = None
    IMPORT_ERROR = 'Install webrtcvad-wheels==2.0.14 in the server Python environment'

CONFIG = dict(engine='WebRTC VAD', mode=2, frame_ms=30,
              start_positive_frames=3, end_negative_frames=10,
              timing='sample-relative; excludes microphone/ALSA/browser latency',
              playback='sampled server playback hints, not acoustic echo cancellation')


class Detector:
    def __init__(self):
        if webrtcvad is None:
            raise RuntimeError(IMPORT_ERROR)
        self.model = webrtcvad.Vad(CONFIG['mode'])
        self.buffer = b''
        self.index = self.positive = self.negative = 0
        self.raw_positive = 0
        self.event = None
        self.events = []
        self.times = []
        self.pending_playback = False
        self.last_end = 0.0

    def feed(self, raw, playback=False):
        self.buffer += raw
        rows = []
        while len(self.buffer) >= 2880:
            frame, self.buffer = self.buffer[:2880], self.buffer[2880:]
            start = self.index * .03
            end = (self.index+1)*.03
            before = time.perf_counter()
            speech = bool(self.model.is_speech(frame, 48000))
            elapsed = (time.perf_counter()-before)*1000
            self.times.append(elapsed)
            self.raw_positive += int(speech)
            if speech:
                if self.positive == 0:
                    self.pending_playback = False
                self.positive += 1
                self.negative = 0
                self.pending_playback |= playback
                self.last_end = end
                if self.event is None and self.positive >= 3:
                    self.event = dict(start_s=round(start-.06,6),
                                      detected_at_s=round(end,6),
                                      playback_possible=self.pending_playback)
            else:
                self.positive = 0
                self.pending_playback = False
                self.negative += 1
            if self.event is not None:
                self.event['playback_possible'] |= playback
                if self.negative >= 10:
                    self.close_event(end, 'silence')
            rows.append(dict(start_s=round(start,6),end_s=round(end,6),
                             speech_raw=speech,speech_active=self.event is not None,
                             playback_possible=bool(playback),process_ms=elapsed))
            self.index += 1
        return rows

    def close_event(self, at, reason):
        if self.event is not None:
            self.event.update(end_s=round(self.last_end,6),
                              closed_at_s=round(at,6),close_reason=reason)
            self.events.append(self.event)
            self.event = None

    def finish(self):
        self.close_event(self.index*.03, 'recording_end')
        return dict(config={**CONFIG,'version':getattr(webrtcvad,'__version__','unknown')},
                    frames=self.index, raw_positive_frames=self.raw_positive,
                    raw_positive_fraction=self.raw_positive/max(self.index,1),
                    unprocessed_tail_samples=len(self.buffer)//2,
                    events=self.events,
                    processing_ms_median=float(np.median(self.times)) if self.times else None,
                    processing_ms_p95=float(np.percentile(self.times,95)) if self.times else None,
                    interpretation='Predicted speech, not confirmed human presence or accuracy. Event intervals can contain short internal gaps. Playback hints are approximate. Raw WAV remains unchanged.')
