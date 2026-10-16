"""HEAR microphone pilot. Install with install(globals()) in server.py."""
import atexit
import csv
import io
import json
import math
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import wave
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import jsonify, request, send_file
from vad_engine import Detector, CONFIG, IMPORT_ERROR

RATE = 48000
BLOCK = 1440


def db(value):
    return round(20 * math.log10(max(float(value), 1e-6)), 2)


def levels(raw):
    x = np.frombuffer(raw, dtype='<i2').astype(np.float64) / 32768
    return dict(rms_dbfs=db(np.sqrt(np.mean(x*x))),
                peak_dbfs=db(np.max(np.abs(x))))


def analyze(path):
    with wave.open(str(path), 'rb') as w:
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype='<i2').astype(np.float64) / 32768
    if not len(x):
        raise ValueError('No audio samples captured')
    n = 4096
    window = np.hanning(n)
    spectra = [np.abs(np.fft.rfft((x[i:i+n]-np.mean(x[i:i+n]))*window))**2
               for i in range(0, len(x)-n+1, n//2)]
    if not spectra:
        raise ValueError('Recording too short for spectrum analysis')
    psd = np.mean(spectra, axis=0) / (RATE * np.sum(window**2))
    psd[1:-1] *= 2
    freq = np.fft.rfftfreq(n, 1/RATE)
    df = RATE/n
    bands = {}
    for lo, hi in [(20,80),(80,250),(250,2000),(2000,8000),(8000,24000)]:
        power = np.sum(psd[(freq >= lo) & (freq < hi)]) * df
        bands[f'{lo}-{hi}_Hz_dbfs'] = db(np.sqrt(power))
    valid = np.where(freq >= 20)[0]
    dominant = float(freq[valid[np.argmax(psd[valid])]]) if np.max(psd) > 0 else None
    result = dict(samples=len(x), duration_s=len(x)/RATE, **levels(raw),
                  dc_offset_fs=float(np.mean(x)),
                  near_full_scale_percent=float(100*np.mean(np.abs(x)>=0.999)),
                  dominant_bin_hz=dominant, spectrum_bin_hz=df, bands=bands)
    result['crest_factor_db'] = round(result['peak_dbfs']-result['rms_dbfs'], 2)
    with path.with_name('spectrum.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frequency_hz','power_fs_squared_per_hz'])
        writer.writerows(zip(freq, psd))
    return result


def install(g):
    app = g['app']
    root = Path(g['HERE']) / 'recordings' / 'hear'
    root.mkdir(parents=True, exist_ok=True)
    for meta_file in root.glob('*/metadata.json'):
        report = meta_file.with_name('report.json')
        if not report.exists() and not meta_file.is_symlink() and not meta_file.parent.is_symlink():
            try:
                interrupted = json.loads(meta_file.read_text(encoding='utf-8'))
                interrupted.update(status='failed', error='Server stopped before finalization; audio may be partial')
                report.write_text(json.dumps(interrupted, indent=2), encoding='utf-8')
            except (OSError, ValueError):
                pass
    mic_lock = threading.Lock()
    state_lock = threading.Lock()
    state = dict(active=False, id=None, elapsed_s=0, level=None, error='', speech=False, events=[], vad_available=not bool(IMPORT_ERROR), vad_error=IMPORT_ERROR)
    processes = []
    original_record = g['record']
    playback_lock = threading.Lock()
    playback_count = 0
    original_play = g.get('play')
    if original_play:
        def tracked_play(path):
            nonlocal playback_count
            with playback_lock:
                playback_count += 1
            try:
                return original_play(path)
            finally:
                with playback_lock:
                    playback_count -= 1
        g['play'] = tracked_play

    def playback_hint():
        with playback_lock:
            playing = playback_count > 0
        # Talk heartbeat and queued data are hints, not proof of acoustic output.
        q = g.get('talk_q')
        return bool(playing or time.time()-g['S'].get('talk_t',0)<1
                    or (q is not None and not q.empty()))

    def guarded_record(sec):
        if not mic_lock.acquire(False):
            return '', subprocess.CompletedProcess([], 1, '', 'HEAR owns microphone; wait for completion')
        try:
            return original_record(sec)
        finally:
            mic_lock.release()
    g['record'] = guarded_record

    def cleanup():
        for proc in list(processes):
            if proc.poll() is None:
                proc.terminate()
    atexit.register(cleanup)

    def update(**kw):
        with state_lock:
            state.update(kw)

    def worker(folder, meta):
        proc = timer = None
        error = ''
        count = 0
        detector = None
        started = time.monotonic()
        try:
            detector = Detector()
            with tempfile.TemporaryFile() as stderr:
                proc = subprocess.Popen(['arecord','-q','-D',meta['device'],
                    '-t','raw','-f','S16_LE','-r',str(RATE),'-c','1',
                    '-d',str(meta['requested_seconds'])], stdout=subprocess.PIPE, stderr=stderr)
                processes.append(proc)
                timer = threading.Timer(meta['requested_seconds']+8, proc.kill)
                timer.daemon = True
                timer.start()
                with wave.open(str(folder/'audio.wav'), 'wb') as wav, (folder/'levels.jsonl').open('w') as log, (folder/'vad_frames.jsonl').open('w') as vad_log:
                    wav.setparams((1,2,RATE,0,'NONE','not compressed'))
                    while True:
                        raw = proc.stdout.read(BLOCK*2)
                        if not raw:
                            break
                        if len(raw)%2:
                            raise ValueError('Incomplete PCM sample')
                        wav.writeframesraw(raw)
                        playback = playback_hint()
                        for prediction in detector.feed(raw, playback):
                            vad_log.write(json.dumps(prediction)+'\n')
                            update(speech_score=prediction['speech_score'])
                        update(speech=detector.event is not None,
                               events=list(detector.events),playback_possible=playback)
                        row = dict(t_s=count/RATE, **levels(raw),
                                   applied_drive=list(g['S'].get('applied_drive', [])), playback_possible=playback)
                        count += len(raw)//2
                        log.write(json.dumps(row)+'\n')
                        update(elapsed_s=round(count/RATE, 2), level=row)
                code = proc.wait(timeout=3)
                stderr.seek(0)
                meta['capture_log'] = stderr.read().decode(errors='replace')[-8000:]
                if code:
                    raise RuntimeError(f'arecord exit {code}: {meta["capture_log"]}')
                if count != meta['requested_seconds']*RATE:
                    raise RuntimeError('Sample count differs from requested duration')
                if meta['capture_log'].strip():
                    meta['warnings'] = ['Inspect capture_log for overruns or ALSA warnings']
            meta['metrics'] = analyze(folder/'audio.wav')
            meta['status'] = 'complete'
        except Exception as exc:
            error = str(exc)
            meta.update(status='failed', error=error)
        finally:
            if timer:
                timer.cancel()
            if proc:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
                proc.stdout.close()
                if proc in processes:
                    processes.remove(proc)
            if detector is not None:
                meta['vad'] = detector.finish()
                update(events=list(detector.events), speech=False)
            meta['samples_saved'] = count
            meta['wall_seconds'] = round(time.monotonic()-started, 3)
            try:
                (folder/'report.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
            except Exception as exc:
                error = f'{error} Report save failed: {exc}'
            finally:
                update(active=False, error=error, speech=False)
                mic_lock.release()

    @app.get('/hear')
    def hear_page():
        return send_file(Path(g['HERE'])/'dashboard'/'hear.html')

    @app.get('/api/hear/status')
    def hear_status():
        records = []
        for f in sorted(root.glob('*/report.json')):
            if f.is_symlink() or f.parent.is_symlink():
                continue
            try:
                records.append(json.loads(f.read_text(encoding='utf-8')))
            except (ValueError, OSError):
                pass
        with state_lock:
            return jsonify(**state, recordings=records)

    @app.post('/api/hear/start')
    def hear_start():
        if IMPORT_ERROR:
            return jsonify(error=IMPORT_ERROR), 503
        j = request.get_json(silent=True) or {}
        scene = str(j.get('scene','')).strip()
        notes = str(j.get('notes',''))[:2000]
        try:
            seconds = int(j.get('seconds',15))
        except (ValueError, TypeError):
            return jsonify(error='Duration must be an integer'), 400
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,60}',scene) or not 5<=seconds<=60:
            return jsonify(error='Scene: letters/digits/_/-. Duration: 5-60 seconds'), 400
        if not mic_lock.acquire(False):
            return jsonify(error='Microphone is busy'), 409
        try:
            dev = g['find_mic']()
            if not dev:
                raise ValueError('No capture device found; check arecord -l')
            if shutil.disk_usage(root).free < 100*1024*1024:
                raise ValueError('Less than 100 MiB free disk space')
            rid = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_')+scene+'_'+uuid.uuid4().hex[:8]
            folder = root/rid
            folder.mkdir()
            meta = dict(id=rid, scene=scene, notes=notes, device=dev,
                requested_seconds=seconds, sample_rate_hz=RATE, channels=1, vad_config=CONFIG,
                sample_format='S16_LE', status='recording', warnings=[],
                started_utc=datetime.now(timezone.utc).isoformat(),
                measurement_note='Digital dBFS, not dB SPL. No software gain or denoising. Device DSP/AGC and ALSA resampling may still apply. Spectrum uses 4096-sample Hann windows, 50% overlap, DC removed per window; bands are unweighted. Near-full-scale samples indicate possible clipping, not proof of all analog clipping.')
            (folder/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
            update(active=True,id=rid,elapsed_s=0,level=None,error='',speech=False,events=[],playback_possible=False,speech_score=None)
            threading.Thread(target=worker,args=(folder,meta),daemon=True).start()
            return jsonify(ok=True,id=rid)
        except Exception as exc:
            update(active=False,error=str(exc))
            mic_lock.release()
            return jsonify(error=str(exc)), 400

    @app.get('/api/hear/download')
    def hear_download():
        # Snapshot only finalized recordings. Active recordings remain untouched.
        folders = [p.parent for p in sorted(root.glob('*/report.json'))
                   if not p.is_symlink() and not p.parent.is_symlink()]
        archive = tempfile.TemporaryFile()
        try:
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                z.writestr('manifest.json',json.dumps(dict(recording_count=len(folders),
                    ids=[p.name for p in folders]),indent=2))
                for folder in folders:
                    for name in ['metadata.json','audio.wav','levels.jsonl','spectrum.csv','vad_frames.jsonl','report.json']:
                        f = folder/name
                        if f.is_file() and not f.is_symlink():
                            z.write(f,folder.name+'/'+name)
            archive.seek(0)
            response = send_file(archive,mimetype='application/zip',as_attachment=True,
                                 download_name='hear_all_'+uuid.uuid4().hex[:8]+'.zip')
            response.call_on_close(archive.close)
            return response
        except Exception:
            archive.close()
            raise
