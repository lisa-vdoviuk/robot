"""HEAR microphone pilot. Install with install(globals()) in server.py."""

import atexit
import csv
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


RATE = 48000
BLOCK = 4800


def db(value):
    return round(
        20 * math.log10(max(float(value), 1e-6)),
        2,
    )


def levels(raw):
    x = (
        np.frombuffer(raw, dtype="<i2").astype(np.float64)
        / 32768
    )

    return {
        "rms_dbfs": db(np.sqrt(np.mean(x * x))),
        "peak_dbfs": db(np.max(np.abs(x))),
    }


def analyze(path):
    with wave.open(str(path), "rb") as wav:
        raw = wav.readframes(wav.getnframes())

    x = (
        np.frombuffer(raw, dtype="<i2").astype(np.float64)
        / 32768
    )

    if not len(x):
        raise ValueError("No audio samples captured")

    n = 4096
    window = np.hanning(n)

    spectra = [
        np.abs(
            np.fft.rfft(
                (x[i:i + n] - np.mean(x[i:i + n]))
                * window
            )
        ) ** 2
        for i in range(0, len(x) - n + 1, n // 2)
    ]

    if not spectra:
        raise ValueError(
            "Recording too short for spectrum analysis"
        )

    psd = (
        np.mean(spectra, axis=0)
        / (RATE * np.sum(window ** 2))
    )

    psd[1:-1] *= 2

    freq = np.fft.rfftfreq(n, 1 / RATE)
    df = RATE / n

    bands = {}

    for lo, hi in [
        (20, 80),
        (80, 250),
        (250, 2000),
        (2000, 8000),
        (8000, 24000),
    ]:
        power = (
            np.sum(psd[(freq >= lo) & (freq < hi)])
            * df
        )

        bands[f"{lo}-{hi}_Hz_dbfs"] = db(
            np.sqrt(power)
        )

    valid = np.where(freq >= 20)[0]

    dominant = (
        float(freq[valid[np.argmax(psd[valid])]])
        if np.max(psd) > 0
        else None
    )

    result = {
        "samples": len(x),
        "duration_s": len(x) / RATE,
        **levels(raw),
        "dc_offset_fs": float(np.mean(x)),
        "near_full_scale_percent": float(
            100 * np.mean(np.abs(x) >= 0.999)
        ),
        "dominant_bin_hz": dominant,
        "spectrum_bin_hz": df,
        "bands": bands,
    }

    result["crest_factor_db"] = round(
        result["peak_dbfs"] - result["rms_dbfs"],
        2,
    )

    with path.with_name("spectrum.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "frequency_hz",
            "power_fs_squared_per_hz",
        ])

        writer.writerows(zip(freq, psd))

    return result


def install(g):
    app = g["app"]

    root = (
        Path(g["HERE"])
        / "recordings"
        / "hear"
    )

    root.mkdir(parents=True, exist_ok=True)

    # Preserve unfinished takes after a server restart.
    for meta_file in root.glob("*/metadata.json"):
        report = meta_file.with_name("report.json")

        if (
            not report.exists()
            and not meta_file.is_symlink()
            and not meta_file.parent.is_symlink()
        ):
            try:
                interrupted = json.loads(
                    meta_file.read_text(
                        encoding="utf-8"
                    )
                )

                interrupted.update(
                    status="failed",
                    error=(
                        "Server stopped before finalization; "
                        "audio may be partial"
                    ),
                )

                report.write_text(
                    json.dumps(
                        interrupted,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            except (OSError, ValueError):
                pass

    mic_lock = threading.Lock()
    state_lock = threading.Lock()

    state = {
        "active": False,
        "id": None,
        "elapsed_s": 0,
        "level": None,
        "error": "",
    }

    processes = []
    original_record = g["record"]

    # Share microphone ownership with existing module tests.
    def guarded_record(sec):
        if not mic_lock.acquire(False):
            return "", subprocess.CompletedProcess(
                [],
                1,
                "",
                (
                    "HEAR owns microphone; "
                    "wait for completion"
                ),
            )

        try:
            return original_record(sec)

        finally:
            mic_lock.release()

    g["record"] = guarded_record

    def cleanup():
        for proc in list(processes):
            if proc.poll() is None:
                proc.terminate()

    atexit.register(cleanup)

    def update(**kwargs):
        with state_lock:
            state.update(kwargs)

    def worker(folder, meta):
        proc = None
        timer = None
        error = ""
        count = 0
        started = time.monotonic()

        try:
            with tempfile.TemporaryFile() as stderr:
                proc = subprocess.Popen(
                    [
                        "arecord",
                        "-q",
                        "-D",
                        meta["device"],
                        "-t",
                        "raw",
                        "-f",
                        "S16_LE",
                        "-r",
                        str(RATE),
                        "-c",
                        "1",
                        "-d",
                        str(meta["requested_seconds"]),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                )

                processes.append(proc)

                timer = threading.Timer(
                    meta["requested_seconds"] + 8,
                    proc.kill,
                )

                timer.daemon = True
                timer.start()

                with (
                    wave.open(
                        str(folder / "audio.wav"),
                        "wb",
                    ) as wav,
                    (folder / "levels.jsonl").open(
                        "w"
                    ) as log,
                ):
                    wav.setparams(
                        (
                            1,
                            2,
                            RATE,
                            0,
                            "NONE",
                            "not compressed",
                        )
                    )

                    while True:
                        raw = proc.stdout.read(
                            BLOCK * 2
                        )

                        if not raw:
                            break

                        if len(raw) % 2:
                            raise ValueError(
                                "Incomplete PCM sample"
                            )

                        wav.writeframesraw(raw)

                        row = {
                            "t_s": count / RATE,
                            **levels(raw),
                            "applied_drive": list(
                                g["S"].get(
                                    "applied_drive",
                                    [],
                                )
                            ),
                        }

                        count += len(raw) // 2

                        log.write(
                            json.dumps(row) + "\n"
                        )

                        update(
                            elapsed_s=round(
                                count / RATE,
                                2,
                            ),
                            level=row,
                        )

                code = proc.wait(timeout=3)

                stderr.seek(0)

                meta["capture_log"] = (
                    stderr.read()
                    .decode(errors="replace")[-8000:]
                )

                if code:
                    raise RuntimeError(
                        f"arecord exit {code}: "
                        f"{meta['capture_log']}"
                    )

                expected_samples = (
                    meta["requested_seconds"] * RATE
                )

                if count != expected_samples:
                    raise RuntimeError(
                        "Sample count differs "
                        "from requested duration"
                    )

                if meta["capture_log"].strip():
                    meta["warnings"] = [
                        "Inspect capture_log for "
                        "overruns or ALSA warnings"
                    ]

            meta["metrics"] = analyze(
                folder / "audio.wav"
            )

            meta["status"] = "complete"

        except Exception as exc:
            error = str(exc)

            meta.update(
                status="failed",
                error=error,
            )

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

            meta["samples_saved"] = count

            meta["wall_seconds"] = round(
                time.monotonic() - started,
                3,
            )

            try:
                (folder / "report.json").write_text(
                    json.dumps(meta, indent=2),
                    encoding="utf-8",
                )

            except Exception as exc:
                error = (
                    f"{error} "
                    f"Report save failed: {exc}"
                )

            finally:
                update(
                    active=False,
                    error=error,
                )

                mic_lock.release()

    @app.get("/hear")
    def hear_page():
        return send_file(
            Path(g["HERE"])
            / "dashboard"
            / "hear.html"
        )

    @app.get("/api/hear/status")
    def hear_status():
        records = []

        for file in sorted(
            root.glob("*/report.json")
        ):
            if (
                file.is_symlink()
                or file.parent.is_symlink()
            ):
                continue

            try:
                records.append(
                    json.loads(
                        file.read_text(
                            encoding="utf-8"
                        )
                    )
                )

            except (ValueError, OSError):
                pass

        with state_lock:
            return jsonify(
                **state,
                recordings=records,
            )

    @app.post("/api/hear/start")
    def hear_start():
        data = (
            request.get_json(silent=True)
            or {}
        )

        scene = str(
            data.get("scene", "")
        ).strip()

        notes = str(
            data.get("notes", "")
        )[:2000]

        try:
            seconds = int(
                data.get("seconds", 15)
            )

        except (ValueError, TypeError):
            return jsonify(
                error="Duration must be an integer"
            ), 400

        if (
            not re.fullmatch(
                r"[A-Za-z0-9_-]{1,60}",
                scene,
            )
            or not 5 <= seconds <= 60
        ):
            return jsonify(
                error=(
                    "Scene: letters/digits/_/-. "
                    "Duration: 5–60 seconds"
                )
            ), 400

        if not mic_lock.acquire(False):
            return jsonify(
                error="Microphone is busy"
            ), 409

        try:
            device = g["find_mic"]()

            if not device:
                raise ValueError(
                    "No capture device found; "
                    "check arecord -l"
                )

            if (
                shutil.disk_usage(root).free
                < 100 * 1024 * 1024
            ):
                raise ValueError(
                    "Less than 100 MiB free disk space"
                )

            recording_id = (
                datetime.now(timezone.utc)
                .strftime("%Y%m%d_%H%M%S_")
                + scene
                + "_"
                + uuid.uuid4().hex[:8]
            )

            folder = root / recording_id
            folder.mkdir()

            meta = {
                "id": recording_id,
                "scene": scene,
                "notes": notes,
                "device": device,
                "requested_seconds": seconds,
                "sample_rate_hz": RATE,
                "channels": 1,
                "sample_format": "S16_LE",
                "status": "recording",
                "warnings": [],
                "started_utc": (
                    datetime.now(timezone.utc)
                    .isoformat()
                ),
                "measurement_note": (
                    "Digital dBFS, not dB SPL. "
                    "No software gain or denoising. "
                    "Device DSP/AGC and ALSA resampling "
                    "may still apply. "
                    "Spectrum uses 4096-sample Hann "
                    "windows, 50% overlap, DC removed "
                    "per window; bands are unweighted. "
                    "Near-full-scale samples indicate "
                    "possible clipping, not proof "
                    "of all analog clipping."
                ),
            }

            (folder / "metadata.json").write_text(
                json.dumps(meta, indent=2),
                encoding="utf-8",
            )

            update(
                active=True,
                id=recording_id,
                elapsed_s=0,
                level=None,
                error="",
            )

            threading.Thread(
                target=worker,
                args=(folder, meta),
                daemon=True,
            ).start()

            return jsonify(
                ok=True,
                id=recording_id,
            )

        except Exception as exc:
            update(
                active=False,
                error=str(exc),
            )

            mic_lock.release()

            return jsonify(
                error=str(exc)
            ), 400

    @app.get("/api/hear/download")
    def hear_download():
        # Active recordings are not included.
        # All finalized takes remain on disk.
        folders = [
            path.parent
            for path in sorted(
                root.glob("*/report.json")
            )
            if (
                not path.is_symlink()
                and not path.parent.is_symlink()
            )
        ]

        archive = tempfile.TemporaryFile()

        try:
            with zipfile.ZipFile(
                archive,
                "w",
                zipfile.ZIP_DEFLATED,
            ) as zip_file:
                zip_file.writestr(
                    "manifest.json",
                    json.dumps(
                        {
                            "recording_count": len(
                                folders
                            ),
                            "ids": [
                                path.name
                                for path in folders
                            ],
                        },
                        indent=2,
                    ),
                )

                for folder in folders:
                    for name in [
                        "metadata.json",
                        "audio.wav",
                        "levels.jsonl",
                        "spectrum.csv",
                        "report.json",
                    ]:
                        file = folder / name

                        if (
                            file.is_file()
                            and not file.is_symlink()
                        ):
                            zip_file.write(
                                file,
                                folder.name + "/" + name,
                            )

            archive.seek(0)

            response = send_file(
                archive,
                mimetype="application/zip",
                as_attachment=True,
                download_name=(
                    "hear_all_"
                    + uuid.uuid4().hex[:8]
                    + ".zip"
                ),
            )

            response.call_on_close(
                archive.close
            )

            return response

        except Exception:
            archive.close()
            raise